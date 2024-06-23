import sys
import load_db
import collections
from langchain_openai import OpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings
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

# Charger la clé JSON depuis Secret Manager
key_json = get_secret('my-service-account-key')
key_data = json.loads(key_json)

# Sauvegarder temporairement la clé pour l'utiliser
with open('/app/service-account-key.json', 'w') as key_file:
    json.dump(key_data, key_file)

# Mettre à jour la variable d'environnement
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/app/service-account-key.json'

class HelpDesk():
    """QA chain"""
    def __init__(self, new_db=True, threshold=0.3):
        self.new_db = new_db
        self.template = self.get_template()
        self.embeddings = self.get_embeddings()
        self.llm = self.get_llm()
        self.prompt = self.get_prompt()
        self.threshold = threshold

        if self.new_db:
            self.db = load_db.DataLoader().set_db(self.embeddings)
        else:
            self.db = load_db.DataLoader().get_db(self.embeddings)

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

    def get_embeddings(self) -> OpenAIEmbeddings:
        embeddings = OpenAIEmbeddings()
        return embeddings

    def get_llm(self):
        llm = OpenAI()
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

if __name__ == "__main__":
    model = HelpDesk(new_db=True, threshold=0.3)

    print(model.db._collection.count())

    prompt = 'Comment est-ce que la formation permet l’obtention de la Certification Professionnelle ?'
    result, sources = model.retrieval_qa_inference(prompt, verbose=False)
    print(result)
    print("youpiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii")
