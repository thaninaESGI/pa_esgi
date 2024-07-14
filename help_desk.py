import json
import os
from flask import Flask, request, jsonify, send_from_directory
import logging
import sys
import load_db
import collections
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
from google.cloud import secretmanager
from google.oauth2 import service_account
from google.cloud import storage  # Import nécessaire pour Google Cloud Storage

# Configurer le logging
logging.basicConfig(level=logging.DEBUG)

# Charger les variables d'environnement depuis le fichier .env
load_dotenv()

# Charger la variable d'environnement SERVICE_ACCOUNT_KEY_JSON depuis le fichier .env
service_account_key_json = os.getenv('SERVICE_ACCOUNT_KEY_JSON')

if not service_account_key_json:
    logging.error("Service account key not found in environment variables")
    sys.exit(1)

def get_secret(secret_id, version_id='latest'):
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{os.getenv('GCP_PROJECT')}/secrets/{secret_id}/versions/{version_id}"
    response = client.access_secret_version(name=name)
    secret = response.payload.data.decode('UTF-8')
    logging.debug(f"Secret fetched: {secret}")
    return secret

def load_service_account_key():
    try:
        secret_data = get_secret('my-service-account-key')
        key_data = json.loads(secret_data)
        logging.debug("Key data successfully loaded from Secret Manager.")
    except json.JSONDecodeError as e:
        logging.error(f"Failed to load key data: {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error retrieving key data: {e}")
        sys.exit(1)

    credentials = service_account.Credentials.from_service_account_info(key_data)
    logging.debug("Credentials created successfully from key data.")
    return credentials

credentials = load_service_account_key()

try:
    openai_api_key = get_secret('openai-api-key')
    if not openai_api_key:
        logging.error("Failed to load OpenAI API Key from Secret Manager.")
        sys.exit(1)
    else:
        logging.debug("OpenAI API Key successfully loaded from Secret Manager.")
except Exception as e:
    logging.error(f"Error retrieving OpenAI API key: {e}")
    sys.exit(1)

# Fonction pour télécharger un fichier depuis GCS
def download_file_from_metadata(res, bucket_name, local_download_path, credentials):
    from google.cloud import storage
    import os

    # Récupérer le nom du fichier à partir des métadonnées
    source = res.metadata.get("source", "Sans source")
    if source == "Sans source":
        return None
    
    # Extraire le nom de fichier uniquement
    file_name = os.path.basename(source)
    local_file_name = os.path.join(local_download_path, file_name.replace(" ", "_"))

    # Télécharger le fichier depuis Google Cloud Storage
    client = storage.Client(credentials=credentials)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(file_name)  # Utiliser uniquement le nom du fichier
    try:
        blob.download_to_filename(local_file_name)
        logging.debug(f"Downloaded {file_name} to {local_file_name}")
    except Exception as e:
        logging.error(f"Erreur lors du téléchargement du fichier {file_name} depuis GCS : {e}")
        return None
    
    return local_file_name

class HelpDesk():
    """QA chain"""
    def __init__(self, new_db=True, threshold=0.3):
        self.new_db = new_db
        self.template = self.get_template()
        self.embeddings = self.get_embeddings(api_key=openai_api_key)
        self.llm = self.get_llm(api_key=openai_api_key)
        self.prompt = self.get_prompt()
        self.threshold = threshold
        self.credentials = credentials
        self.db_version = 0  # Version de la base de données
        self.bucket_name = 'ingestion_bucket_1'
        self.backend_base_url = 'https://help-desk-service-beta-lxazwit43a-od.a.run.app'  # Remplacez par l'URL de votre backend

        # Initialiser la base de données au démarrage
        self.initialize_db()

    def initialize_db(self):
        try:
            self.db_path = f'/tmp/db/chroma_v{self.db_version}/'
            os.makedirs(self.db_path, exist_ok=True)
            os.chmod(self.db_path, 0o777)
            
            if self.new_db:
                self.db = load_db.DataLoader(credentials=self.credentials, persist_directory=self.db_path).set_db(self.embeddings)
            else:
                self.db = load_db.DataLoader(credentials=self.credentials, persist_directory=self.db_path).get_db(self.embeddings)

            self.retriever = self.db.as_retriever()
            self.retrieval_qa_chain = self.get_retrieval_qa()
        except Exception as e:
            logging.error(f"Error initializing database: {e}")
            raise

    def get_template(self):
        template = """
        Etant donnés ces textes:
        -----
        {context}
        -----
        Répond à la question suivante:
        Question: {question}
        Réponse utile:
        """
        return template

    def get_prompt(self) -> PromptTemplate:
        prompt = PromptTemplate(
            template=self.template,
            input_variables=["context", "question"]
        )
        return prompt

    def get_embeddings(self, api_key) -> OpenAIEmbeddings:
        embeddings = OpenAIEmbeddings(api_key=api_key)
        return embeddings

    def get_llm(self, api_key):
        llm = OpenAI(api_key=api_key)
        return llm

    def get_retrieval_qa(self):
        chain_type_kwargs = {"prompt": self.prompt}
        qa = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True,
            chain_type_kwargs=chain_type_kwargs
        )
        return qa

    def retrieval_qa_inference(self, question, verbose=True):
        query = {"query": question}
        answer = self.retrieval_qa_chain(query)
        sources = self.list_top_k_sources(answer, k=2)

        if verbose:
            print(sources)

        return answer["result"], sources

    def list_top_k_sources(self, answer, k=2):
        sources = []
        local_download_path = '/tmp/pdf_downloads'
        os.makedirs(local_download_path, exist_ok=True)
        
        for res in answer["source_documents"]:
            source = res.metadata.get("source", "Sans source")
            if source != "Sans source":
                title = res.metadata.get("title", os.path.basename(source))
                local_file_name = download_file_from_metadata(res, self.bucket_name, local_download_path, self.credentials)
                if local_file_name:
                    # Construire le lien vers le fichier avec l'URL absolue
                    file_link = f"{self.backend_base_url}/files/{os.path.basename(local_file_name)}"
                    sources.append(f'[{title}]({file_link})')

        if sources:
            k = min(k, len(sources))
            distinct_sources = list(zip(*collections.Counter(sources).most_common()))[0][:k]
            distinct_sources_str = "  \n- ".join(distinct_sources)

            if len(distinct_sources) == 1:
                return f"Voici la source qui pourrait t'être utile :  \n- {distinct_sources_str}"

            elif len(distinct_sources) > 1:
                return f"Voici {len(distinct_sources)} sources qui pourraient t'être utiles :  \n- {distinct_sources_str}"

        return "Je n'ai pas trouvé de ressource pour répondre à ta question"

app = Flask(__name__)
model = HelpDesk(new_db=True, threshold=0.3)

@app.route('/', methods=['GET'])
def home():
    return "Service is running", 200

@app.route('/reload', methods=['POST'])
def reload():
    try:
        model.reload_data()
        logging.debug("Data reloaded successfully.")
        return jsonify({"status": "success", "message": "Data reloaded successfully."}), 200
    except Exception as e:
        logging.error(f"Error reloading data: {e}")
        return jsonify({"status": "error", "message": "Error reloading data."}), 500

@app.route('/', methods=['POST'])
def query():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data received"}), 400

    question = data.get("question", "")
    if not question:
        return jsonify({"error": "No question provided"}), 400

    try:
        result, sources = model.retrieval_qa_inference(question, verbose=False)
    except Exception as e:
        logging.error(f"Error during inference: {e}")
        return jsonify({"error": "Error during inference"}), 500

    response = {
        "result": result,
        "sources": sources
    }
    return jsonify(response)

# Route pour servir les fichiers téléchargés
@app.route('/files/<path:filename>', methods=['GET'])
def serve_file(filename):
    return send_from_directory('/tmp/pdf_downloads', filename)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)
