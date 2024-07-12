import logging
import shutil
from google.cloud import storage
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter

class DataLoader():
    """Create, load, save the DB using the PDF Loader"""
    def __init__(
        self,
        credentials,
        directories=['ingestion_bucket_1'],
        persist_directory='/tmp/db/chroma/',  # Utilisez un répertoire accessible en écriture
        bucket_name='ingestion_bucket_1'
    ):
        self.directories = directories
        self.persist_directory = persist_directory
        self.bucket_name = bucket_name
        self.credentials = credentials
        self.db = None  # Initialize the db attribute

    def download_pdfs_from_bucket(self):
        """Download PDF files from Google Cloud Storage bucket"""
        client = storage.Client(credentials=self.credentials)
        bucket = client.bucket(self.bucket_name)
        blobs = bucket.list_blobs()

        local_paths = []
        for blob in blobs:
            if blob.name.endswith('.pdf'):
                local_path = f'/tmp/{blob.name}'
                blob.download_to_filename(local_path)
                local_paths.append(local_path)
        return local_paths

    def load_from_pdf_loader(self):
        """Load PDF files from specified directories"""
        docs = []
        pdf_files = self.download_pdfs_from_bucket()
        for pdf_file in pdf_files:
            loader = PyPDFLoader(pdf_file)
            loaded_docs = loader.load()
            for doc in loaded_docs:
                doc.metadata['source'] = str(pdf_file)  # Ajoute le nom de fichier aux métadonnées
            docs.extend(loaded_docs)
        return docs

    def split_docs(self, docs):
        # Markdown splitting settings
        headers_to_split_on = [
            ("#", "Titre 1"),
            ("##", "Sous-titre 1"),
            ("###", "Sous-titre 2"),
        ]

        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

        # Split based on markdown and add original metadata
        md_docs = []
        for doc in docs:
            md_doc = markdown_splitter.split_text(doc.page_content)
            for i in range(len(md_doc)):
                md_doc[i].metadata = md_doc[i].metadata | doc.metadata
            md_docs.extend(md_doc)

        # Split into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=20,
            separators=["\n\n", "\n", "(?<=\. )", " ", ""]
        )

        splitted_docs = splitter.split_documents(md_docs)

        # Add context to chunks
        enriched_docs = []
        for i, doc in enumerate(splitted_docs):
            if i > 0:
                doc.metadata["previous_chunk"] = splitted_docs[i-1].page_content
            if i < len(splitted_docs) - 1:
                doc.metadata["next_chunk"] = splitted_docs[i+1].page_content
            enriched_docs.append(doc)

        return enriched_docs

    def save_to_db(self, splitted_docs, embeddings):
        batch_size = 50 # Choisissez une taille de lot appropriée
        db = None
        for i in range(0, len(splitted_docs), batch_size):
            batch = splitted_docs[i:i + batch_size]
            if db is None:
                db = Chroma.from_documents(batch, embeddings, persist_directory=self.persist_directory)
            else:
                db.add_documents(batch)
        if db:
            db.persist()
        return db

    def load_from_db(self, embeddings):
        """Load chunks from Chroma DB"""
        db = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=embeddings
        )
        self.db = db  # Set the db attribute
        return db

    def set_db(self, embeddings):
        """Create, save, and load db"""
        try:
            shutil.rmtree(self.persist_directory)
        except Exception as e:
            logging.warning("%s", e)

        # Load docs
        docs = self.load_from_pdf_loader()

        # Split docs and add context
        splitted_docs = self.split_docs(docs)

        db = self.save_to_db(splitted_docs, embeddings)

        return db

    def get_db(self, embeddings):
        """Create, save, and load db"""
        db = self.load_from_db(embeddings)
        return db
