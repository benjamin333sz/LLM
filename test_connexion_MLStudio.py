import requests

try:
    response = requests.get("http://127.0.0.1:1234/v1/models/", timeout=5)
    print("Connexion OK:", response.status_code)
    print(response.json())
except Exception as e:
    print(f"Erreur de connexion: {e}")