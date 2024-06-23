# Utilisez une image de base Python
FROM python:3.10-slim

# Définissez le répertoire de travail dans le conteneur
WORKDIR /app

# Copiez les fichiers de requirements dans le répertoire de travail
COPY requirements.txt .

# Installez les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiez les fichiers du projet dans le répertoire de travail
COPY . .

# Définir la variable d'environnement pour le fichier des credentials
ENV GOOGLE_APPLICATION_CREDENTIALS=/app/pa-ingestion-98640c36406b.json
COPY pa-ingestion-98640c36406b.json  /app/pa-ingestion-98640c36406b.json

# Copiez le fichier .env dans le conteneur
COPY .env .env

# Exécutez le script principal
CMD ["python", "help_desk.py"]

