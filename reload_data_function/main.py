# main.py
import base64
import requests
import os

def reload_data(event, context):
    request_json = event.get('data')
    if request_json:
        pubsub_message = base64.b64decode(request_json).decode('utf-8')
        print(f"Pub/Sub message: {pubsub_message}")
        
        # Appeler le point de terminaison de rechargement de données de help_dsk
        help_dsk_reload_url = os.getenv('HELP_DSK_RELOAD_URL')
        response = requests.post(help_dsk_reload_url)
        
        if response.status_code == 200:
            return 'Data reload successful', 200
        else:
            return 'Failed to reload data in help_desk', 500
    return 'No Pub/Sub message received', 400

