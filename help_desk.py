from flask import Flask, request, jsonify
import sys
import load_db
import collections
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
import os
import json
from google.cloud import secretmanager

# Charger les variables d'environnement depuis le fichier .env
load_dotenv()

def get_secret(secret_id, version_id='latest'):
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{os.getenv('GCP_PROJECT')}/secrets/{secret_id}/versions/{version_id}"
    response = client.access_secret_version(name=name)
    secret = response.payload.data.decode('UTF-8')
    return secret

# Charger la clé JSON depuis Secret Manager via la variable d'environnement
key_json = os.getenv('SERVICE_ACCOUNT_KEY_JSON')
if key_json is None:
    print("Environment variable SERVICE_ACCOUNT_KEY_JSON is not set.")
    sys.exit(1)

try:
    key_data = json.loads(key_json)
    print("Key data successfully loaded.")
except json.JSONDecodeError as e:
    print("Failed to load key data:", e)
    sys.exit(1)

# Définir le chemin du fichier de clé JSON
credentials_path = '/app/service-account-key.json'

# Sauvegarder temporairement la clé pour l'utiliser
try:
    with open(credentials_path, 'w') as key_file:
        json.dump(key_data, key_file)
    print("Key file written successfully.")
except IOError as e:
    print("Failed to write key file:", e)
    sys.exit(1)

# Mettre à jour la variable d'environnement
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
print("Environment variable set successfully.")

# Récupérer la clé API OpenAI depuis Secret Manager
openai_api_key = get_secret('openai-api-key')
if not openai_api_key:
    print("Failed to load OpenAI API Key from Secret Manager.")
    sys.exit(1)
else:
    print("OpenAI API Key successfully loaded from Secret Manager.")

class HelpDesk():
    """QA chain"""
    def __init__(self, new_db=True, threshold=0.3):
        self.new_db = new_db
        self.template = self.get_template()
        self.embeddings = self.get_embeddings(api_key=openai_api_key)
        self.llm = self.get_llm(api_key=openai_api_key)
        self.prompt = self.get_prompt()
        self.threshold = threshold

        # Passer le chemin des credentials à DataLoader
        if self.new_db:
            self.db = load_db.DataLoader(credentials_path=credentials_path).set_db(self.embeddings)
        else:
            self.db = load_db.DataLoader(credentials_path=credentials_path).get_db(self.embeddings)

        self.retriever = self.db.as_retriever()
        self.retrieval_qa_chain = self.get_retrieval_qa()

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
        filtered_answer = self.filter_by_similarity(answer, self.threshold)
        sources = self.list_top_k_sources(filtered_answer, k=3)

        if verbose:
            print(sources)

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

        return "Je n'ai trouvé pas trouvé de ressource pour répondre à ta question"

# Créer une instance de Flask
app = Flask(__name__)

# Charger le modèle HelpDesk
model = HelpDesk(new_db=True, threshold=0.3)

@app.route('/query', methods=['POST'])
def query():
    data = request.get_json()
    question = data.get("question", "")
    if not question:
        return jsonify({"error": "No question provided"}), 400

    result, sources = model.retrieval_qa_inference(question, verbose=False)
    response = {
        "result": result,
        "sources": sources
    }
    return jsonify(response)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
