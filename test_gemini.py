import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    print("[ERREUR] Clé API introuvable. Vérifie ton fichier .env")
    exit()

def tester_cerveau_rest():
    print("Envoi de la requête à Gemini via l'API REST...")
    
    # L'URL mise à jour pour cibler la génération actuelle (gemini-3.1-flash-lite)
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.1-flash-lite:generateContent?key={API_KEY}"
    headers = {'Content-Type': 'application/json'}
    
    consigne = """Tu es le cerveau d'un petit robot. 
    Réponds à cette phrase de l'utilisateur uniquement avec ce format JSON strict, sans aucun autre texte autour :
    {"texte": "ta réponse parlée courte", "emotion": "happy"}
    
    Phrase de l'utilisateur : Bonjour, qui es-tu ?"""
    
    payload = {
        "contents": [{"parts": [{"text": consigne}]}]
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status() 
        
        donnees_api = response.json()
        texte_brut = donnees_api['candidates'][0]['content']['parts'][0]['text']
        
        texte_brut = texte_brut.replace('```json', '').replace('```', '').strip()
        data = json.loads(texte_brut)
        
        print("\n[SUCCÈS] JSON valide reçu :")
        print(f"- Le robot dira : {data['texte']}")
        print(f"- L'animation sera : {data['emotion']}\n")
        
    except requests.exceptions.HTTPError as err:
        print(f"\n[ERREUR HTTP] {err}")
        print(response.text)
    except Exception as e:
        print(f"\n[ERREUR] {e}")

if __name__ == "__main__":
    tester_cerveau_rest()