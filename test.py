from dotenv import load_dotenv
import os

load_dotenv()

# Vérifiez que la variable d'environnement est correctement chargée
print(os.getenv("GOOGLE_APPLICATION_CREDENTIALS"))

