from google.cloud import storage
import json
import os

# Charger les identifiants depuis le fichier JSON généré dynamiquement
try:
    key_json = os.getenv('SERVICE_ACCOUNT_KEY_JSON')
    if key_json is None:
        raise ValueError("Environment variable SERVICE_ACCOUNT_KEY_JSON is not set.")

    key_data = json.loads(key_json)
except json.JSONDecodeError as e:
    print(f"Failed to load key data: {e}")
    sys.exit(1)

# Sauvegarder temporairement les identifiants dans un fichier
credentials_path = '/tmp/service-account-key.json'
try:
    with open(credentials_path, 'w') as key_file:
        json.dump(key_data, key_file)
except IOError as e:
    print(f"Failed to write key file: {e}")
    sys.exit(1)

# Mettre à jour la variable d'environnement pour utiliser les identifiants
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path

# Testez l'accès à Google Cloud Storage
try:
    client = storage.Client()
    buckets = list(client.list_buckets())
    print("Buckets:")
    for bucket in buckets:
        print(bucket.name)
except Exception as e:
    print(f"Erreur: {e}")
