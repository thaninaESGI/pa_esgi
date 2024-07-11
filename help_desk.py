import json
import os
from flask import Flask, request, jsonify
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

        # Initialiser la base de données au démarrage
        self.initialize_db()

    def initialize_db(self):
        try:
            if self.new_db:
                self.db = load_db.DataLoader(credentials=self.credentials, persist_directory='/tmp/db/chroma/').set_db(self.embeddings)
            else:
                self.db = load_db.DataLoader(credentials=self.credentials, persist_directory='/tmp/db/chroma/').get_db(self.embeddings)

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
        try:
            answer = self.retrieval_qa_chain(query)
        except Exception as e:
            logging.error(f"Error during retrieval QA chain: {e}")
            raise e
        filtered_answer = self.filter_by_similarity(answer, self.threshold)
        sources = self.list_top_k_sources(filtered_answer, k=3)
        return filtered_answer["result"], sources

    def filter_by_similarity(self, answer, threshold):
        filtered_docs = []
        for doc in answer["source_documents"]:
            if doc.metadata.get("score", 0) > threshold:
                filtered_docs.append(doc)
        answer["source_documents"] = filtered_docs
        return answer

    def list_top_k_sources(self, answer, k=3):
        sources = [
            f'[{res.metadata.get("title", "Sans titre")}]({res.metadata.get("source", "Sans source")})'
            for res in answer["source_documents"]
        ]

        if sources:
            k = min(k, len(sources))
            distinct_sources = list(zip(*collections.Counter(sources).most_common()))[0][:k]
            distinct_sources_str = "  \n- ".join(distinct_sources)

            if len(distinct_sources) == 1:
                return f"Voici la source qui pourrait t'être utile :  \n- {distinct_sources_str}"

            elif len(distinct_sources) > 1:
                return f"Voici {len(distinct_sources)} sources qui pourraient t'être utiles :  \n- {distinct_sources_str}"
        else:
            return "Je n'ai trouvé pas trouvé de ressource pour répondre à ta question"

    def reload_data(self):
        logging.debug("Reloading data from cloud storage...")
        try:
            self.db = load_db.DataLoader(credentials=self.credentials, persist_directory='/tmp/db/chroma/').set_db(self.embeddings)
            self.retriever = self.db.as_retriever()
            self.retrieval_qa_chain = self.get_retrieval_qa()
            logging.debug("Data reload completed successfully.")
        except Exception as e:
            logging.error(f"Error reloading data: {e}")
            raise

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

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)
