import os
import json
import sys
from google.cloud import secretmanager, storage
from dotenv import load_dotenv

# Charger les variables d'environnement depuis le fichier .env
load_dotenv()

def get_secret(secret_id, version_id='latest'):
    client = secretmanager.SecretManagerServiceClient()
    project_id = os.getenv('GCP_PROJECT')
    if not project_id:
        print("Environment variable GCP_PROJECT is not set.")
        sys.exit(1)
    name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"
    response = client.access_secret_version(name=name)
    secret = response.payload.data.decode('UTF-8')
    return secret

# Charger la clé JSON depuis Secret Manager via la variable d'environnement
key_json = get_secret('my-service-account-key')
if key_json is None:
    print("Failed to retrieve the service account key from Secret Manager.")
    sys.exit(1)

try:
    key_data = json.loads(key_json)
    print("Key data successfully loaded.")
except json.JSONDecodeError as e:
    print("Failed to load key data:", e)
    sys.exit(1)

# Définir le chemin du fichier de clé JSON
credentials_path = '/tmp/service-account-key.json'

# Sauvegarder temporairement la clé pour l'utiliser
try:
    with open(credentials_path, 'w') as key_file:
        json.dump(key_data, key_file)
    print("Key file written successfully.")
except IOError as e:
    print("Failed to write key file:", e)
    sys.exit(1)

# Mettre à jour la variable d'environnement pour utiliser les identifiants
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
print("Environment variable set successfully.")

# Testez l'accès à Google Cloud Storage
try:
    client = storage.Client()
    buckets = list(client.list_buckets())
    print("Buckets:")
    for bucket in buckets:
        print(bucket.name)
except Exception as e:
    print(f"Erreur: {e}")
