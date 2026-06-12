"""
main.py — Cozmo IA : Assistant multimodal + Waypoint Navigation
===============================================================
Architecture modulaire :
  • Mode Conversationnel : voix/clavier + caméra + Gemini + actions physiques
  • Mode Météo OLED    : widget graphique 128x64 sur le visage de Cozmo
  • Mode Exploration    : Waypoint Navigation Chain of Thought (→ navigation.py)

Python 3.8 strict — toutes les tâches bloquantes via loop.run_in_executor.
"""

import cozmo
import cozmo.oled_face
import os
import json
import requests
import asyncio
import functools
import base64
import io
import math
import speech_recognition as sr
from PIL import Image, ImageDraw, ImageFont
from dotenv import load_dotenv

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# Module de navigation Waypoint (Stop, Look, Think, Act + Fail-Safe)
from navigation import lancer_exploration

# =====================================================================
# --- CONFIGURATION ---
# =====================================================================
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

MAX_HISTORIQUE = 6          # 3 tours user/model max dans la mémoire FIFO
historique_conversation = []

# --- SERVEUR WEB & WEBSOCKET ---
app = FastAPI()
app.mount("/frontend", StaticFiles(directory="frontend", html=True), name="frontend")

active_websockets = []
command_queue = asyncio.Queue()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_websockets.append(websocket)
    try:
        while True:
            data = await websocket.receive_json()
            if data.get("type") == "command":
                await command_queue.put(data.get("command"))
    except WebSocketDisconnect:
        active_websockets.remove(websocket)

async def broadcast_data(data: dict):
    for ws in list(active_websockets):
        try:
            await ws.send_json(data)
        except Exception:
            pass

async def start_server():
    config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="warning")
    server = uvicorn.Server(config)
    await server.serve()

async def stream_video(robot):
    while True:
        await asyncio.sleep(0.1)  # ~10 FPS
        try:
            img_b64 = capturer_image_sync(robot)
            if img_b64:
                await broadcast_data({"type": "video_frame", "frame": img_b64})
        except Exception:
            pass

async def stream_telemetry(robot):
    while True:
        await asyncio.sleep(0.1)
        try:
            if hasattr(robot, 'pose'):
                await broadcast_data({
                    "type": "state_update",
                    "odometry": {
                        "x": robot.pose.position.x,
                        "y": robot.pose.position.y,
                        "cap": robot.pose.rotation.angle_z.degrees
                    }
                })
        except Exception:
            pass

async def terminal_input_task(loop):
    while True:
        try:
            cmd = await loop.run_in_executor(None, input, "⌨️ Commande Web/Terminal (ou Entrée pour micro) : \n")
            if not cmd.strip():
                cmd = await loop.run_in_executor(None, capturer_voix_sync)
            if cmd:
                await command_queue.put(cmd)
        except Exception:
            pass
# =====================================================================
# --- LES YEUX (Capture caméra → base64) ---
# =====================================================================
def capturer_image_sync(robot):
    """Capture la dernière image Cozmo, redimensionne et encode en base64."""
    latest = robot.world.latest_image
    if latest is None or latest.raw_image is None:
        return None

    pil_image = latest.raw_image.resize((320, 240))

    buffer = io.BytesIO()
    pil_image.save(buffer, format="JPEG", quality=75)
    image_bytes = buffer.getvalue()
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    return image_b64


# =====================================================================
# --- LE CERVEAU (Gemini conversationnel — texte + image + mémoire) ---
# =====================================================================
async def demander_au_cerveau(question, image_b64=None):
    """
    Envoie la question + image optionnelle à Gemini avec l'historique FIFO.
    Retourne (réponse_json, texte_brut).
    """
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-3.1-flash-lite:generateContent?key={API_KEY}"
    )
    headers = {'Content-Type': 'application/json'}

    system_prompt = (
        "Tu es Cozmo, un petit robot de bureau malicieux et très intelligent. "
        "L'utilisateur va te parler et tu peux aussi VOIR ce qu'il y a devant toi "
        "grâce à ta caméra frontale. "
        "Tu dois répondre UNIQUEMENT avec ce format JSON strict :\n"
        '{"texte": "ta réponse parlée courte (maximum 2 phrases)", '
        '"emotion": "happy|sad|surprised|angry", '
        '"action": "none|lever_bras|baisser_bras|avancer|reculer|tourner_gauche|tourner_droite", '
        '"ville_meteo": "nom de la ville ou null", '
        '"valeur": "nombre ou null"}\n\n'
        "Règles pour l'action :\n"
        "- Choisis UNE SEULE action physique logique.\n"
        "- Utilise 'none' si aucune action physique n'est pertinente.\n\n"
        "Règles pour valeur :\n"
        "- Si l'utilisateur précise une distance (ex: 'avance de 15 cm'), "
        "mets le nombre en centimètres (ex: 15).\n"
        "- Si l'utilisateur précise un angle (ex: 'tourne de 180 degrés'), "
        "mets le nombre en degrés (ex: 180).\n"
        "- Si aucune valeur n'est mentionnée, renvoie null.\n\n"
        "Règles pour ville_meteo :\n"
        "- Si l'utilisateur demande la météo d'une ville, remplis avec le nom.\n"
        "- Sinon, renvoie null."
    )

    user_parts = [{"text": question}]
    if image_b64 is not None:
        user_parts.append({
            "inlineData": {"mimeType": "image/jpeg", "data": image_b64}
        })

    message_courant = {"role": "user", "parts": user_parts}
    payload = {
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": historique_conversation + [message_courant]
    }


    loop = asyncio.get_event_loop()
    requete = functools.partial(requests.post, url, headers=headers, json=payload)

    # --- Retry avec backoff exponentiel (429 Too Many Requests) ---
    max_tentatives = 3
    delai = 2  # secondes, doublé à chaque tentative

    for tentative in range(1, max_tentatives + 1):
        response = await loop.run_in_executor(None, requete)
        try:
            response.raise_for_status()
            break  # Succès, on sort de la boucle
        except requests.exceptions.HTTPError as e:
            if response.status_code == 429 and tentative < max_tentatives:
                pass  # Retry silencieux
                await asyncio.sleep(delai)
                delai *= 2
            else:
                raise  # Autre erreur ou dernière tentative → on propage

    donnees = response.json()
    texte_brut = donnees['candidates'][0]['content']['parts'][0]['text']
    texte_brut = texte_brut.replace('```json', '').replace('```', '').strip()

    return json.loads(texte_brut), texte_brut


# =====================================================================
# --- LES OREILLES (Micro) ---
# =====================================================================
def capturer_voix_sync():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source, duration=1.0)
        print("🎤 Parle maintenant...")
        try:
            audio = r.listen(source, timeout=5, phrase_time_limit=10)
            texte = r.recognize_google(audio, language="fr-FR")
            return texte
        except sr.WaitTimeoutError:
            return ""
        except sr.UnknownValueError:
            return ""
        except Exception:
            return ""


# =====================================================================
# --- L'ÉCRAN (Widget Météo OLED 128x64) ---
# =====================================================================
def obtenir_meteo_sync(ville):
    """Interroge wttr.in, retourne (temp_str, condition_str) ou (None, None)."""
    try:
        resp = requests.get(f"https://wttr.in/{ville}?format=j1", timeout=5)
        resp.raise_for_status()
        data = resp.json()
        current = data["current_condition"][0]
        temp = current["temp_C"]
        condition = current["weatherDesc"][0]["value"]
        return temp, condition
    except Exception:
        return None, None


def _dessiner_soleil(draw, cx, cy, r):
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=1)
    for a in range(0, 360, 45):
        rad = math.radians(a)
        x1 = cx + int((r + 2) * math.cos(rad))
        y1 = cy + int((r + 2) * math.sin(rad))
        x2 = cx + int((r + 6) * math.cos(rad))
        y2 = cy + int((r + 6) * math.sin(rad))
        draw.line([x1, y1, x2, y2], fill=1, width=1)


def _dessiner_nuage(draw, cx, cy):
    draw.ellipse([cx - 10, cy - 4, cx + 2, cy + 6], fill=1)
    draw.ellipse([cx - 4, cy - 8, cx + 8, cy + 4], fill=1)
    draw.ellipse([cx + 2, cy - 4, cx + 14, cy + 6], fill=1)
    draw.rectangle([cx - 8, cy + 1, cx + 12, cy + 6], fill=1)


def _dessiner_pluie(draw, cx, cy):
    _dessiner_nuage(draw, cx, cy - 4)
    for dx in [-6, 0, 6]:
        draw.line([cx + dx, cy + 4, cx + dx - 2, cy + 10], fill=1, width=1)


def _dessiner_neige(draw, cx, cy):
    _dessiner_nuage(draw, cx, cy - 4)
    for dx in [-6, 0, 6]:
        draw.ellipse([cx + dx - 1, cy + 5, cx + dx + 1, cy + 7], fill=1)
        draw.ellipse([cx + dx + 2, cy + 9, cx + dx + 4, cy + 11], fill=1)


def generer_image_meteo(ville, temp, condition):
    """Génère une image PIL 128x64 monochrome pour l'OLED de Cozmo."""
    img = Image.new("1", (128, 64), color=0)
    draw = ImageDraw.Draw(img)

    try:
        font_large = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        font_small = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
    except (IOError, OSError):
        font_large = ImageFont.load_default()
        font_small = ImageFont.load_default()

    # Ville en haut, centrée
    ville_txt = ville[:12].upper()
    bbox = draw.textbbox((0, 0), ville_txt, font=font_small)
    draw.text(((128 - (bbox[2] - bbox[0])) // 2, 1), ville_txt,
              fill=1, font=font_small)

    draw.line([4, 13, 124, 13], fill=1)

    # Température
    draw.text((6, 20), f"{temp} C", fill=1, font=font_large)

    # Icône météo
    cond = condition.lower()
    cx, cy = 100, 36
    if "sun" in cond or "clear" in cond:
        _dessiner_soleil(draw, cx, cy, 8)
    elif "rain" in cond or "drizzle" in cond or "shower" in cond:
        _dessiner_pluie(draw, cx, cy)
    elif "snow" in cond or "sleet" in cond or "ice" in cond:
        _dessiner_neige(draw, cx, cy)
    elif "cloud" in cond or "overcast" in cond or "fog" in cond or "mist" in cond:
        _dessiner_nuage(draw, cx, cy)
    else:
        draw.text((cx - 4, cy - 6), "?", fill=1, font=font_large)

    # Condition en bas
    cond_txt = condition[:18]
    bbox_c = draw.textbbox((0, 0), cond_txt, font=font_small)
    draw.text(((128 - (bbox_c[2] - bbox_c[0])) // 2, 52), cond_txt,
              fill=1, font=font_small)


    return img


# =====================================================================
# --- LE CORPS (Boucle principale) ---
# =====================================================================
async def cozmo_program(robot: cozmo.robot.Robot):
    robot.camera.image_stream_enabled = True
    await asyncio.sleep(1.0)
    await robot.set_head_angle(cozmo.util.degrees(30)).wait_for_completed()
    print("⚙️ Connexion établie.")
    loop = asyncio.get_event_loop()

    # Démarrage des tâches de fond
    asyncio.create_task(start_server())
    asyncio.create_task(stream_video(robot))
    asyncio.create_task(stream_telemetry(robot))
    asyncio.create_task(terminal_input_task(loop))
    
    await broadcast_data({"type": "terminal_log", "message": "Système prêt. Entrez une commande.", "source": "system"})

    while True:
        question = await command_queue.get()
        print(f"⌨️ Commande reçue : {question.strip()}")
        await broadcast_data({"type": "terminal_log", "message": question.strip(), "source": "user"})

        # --- Commandes de sortie ---
        if question.lower() in ["quitter", "exit", "stop", "arrête-toi"]:
            await robot.say_text(
                "Désactivation vocale en cours. Au revoir.",
                in_parallel=True
            ).wait_for_completed()
            break

        # --- Détection du mode exploration (Waypoint Navigation) ---
        # Liste de mots-clés déclencheurs pour activer la navigation.
        mots_cles_nav = [
            "va dans", "va vers", "va à", "va au",
            "explore", "dirige", "rejoins", "trouve"
        ]
        question_lower = question.lower()

        mot_detecte = None
        for mot in mots_cles_nav:
            if mot in question_lower:
                mot_detecte = mot
                break

        if mot_detecte:
            commande_complete = question.strip()

            await robot.say_text("Mode exploration activé.", in_parallel=True).wait_for_completed()
            await broadcast_data({"type": "terminal_log", "message": "Mode exploration activé.", "source": "system"})

            # Lancement de l'exploration (en lui passant broadcast_data pour logger)
            await lancer_exploration(robot, commande_complete, broadcast_data)
            continue

        # --- Capture image caméra ---
        try:
            image_b64 = await loop.run_in_executor(
                None, functools.partial(capturer_image_sync, robot)
            )
        except Exception:
            image_b64 = None

        print("🧠 Réfléchis...")
        robot.set_all_backpack_lights(cozmo.lights.blue_light.flash())

        try:
            # Appel Gemini conversationnel
            reponse_ia, texte_brut_modele = await demander_au_cerveau(
                question, image_b64
            )

            # Mise à jour mémoire FIFO
            historique_conversation.append(
                {"role": "user", "parts": [{"text": question}]}
            )
            historique_conversation.append(
                {"role": "model", "parts": [{"text": texte_brut_modele}]}
            )
            while len(historique_conversation) > MAX_HISTORIQUE:
                historique_conversation.pop(0)

            texte = reponse_ia.get("texte", "J'ai un trou de mémoire.")
            emotion = reponse_ia.get("emotion", "happy")
            action = reponse_ia.get("action", "none")
            ville = reponse_ia.get("ville_meteo", None)
            val_display = reponse_ia.get("valeur", "")

            # Log vers le Dashboard
            await broadcast_data({
                "type": "ai_cot",
                "etape": "Conversation",
                "diagnostic": texte,
                "action": action,
                "valeur": val_display,
                "raw": reponse_ia,
                "odometry": {
                    "x": robot.pose.position.x if robot.pose else 0,
                    "y": robot.pose.position.y if robot.pose else 0,
                    "cap": robot.pose.rotation.angle_z.degrees if robot.pose else 0
                },
                "frame": capturer_image_sync(robot)
            })
            await broadcast_data({"type": "terminal_log", "message": texte, "source": "robot"})

            print(f"🤖 Cozmo : {texte}")
            if action != "none":
                if val_display:
                    print(f"🏎️ Action : {action} de {val_display}")
                else:
                    print(f"🏎️ Action : {action}")

            # --- Animation faciale ---
            anim = cozmo.anim.Triggers.CodeLabHappy
            if emotion == "sad":
                anim = cozmo.anim.Triggers.CodeLabDejected
            elif emotion == "surprised":
                anim = cozmo.anim.Triggers.CodeLabSurprise
            elif emotion == "angry":
                anim = cozmo.anim.Triggers.CodeLabFrustrated

            action_parole = robot.say_text(texte, in_parallel=True)
            action_anim = robot.play_anim_trigger(anim, in_parallel=True)

            # --- Routeur d'actions physiques ---
            action_physique = None
            val = reponse_ia.get("valeur")
            if action == "lever_bras":
                action_physique = robot.set_lift_height(1.0, in_parallel=True)
            elif action == "baisser_bras":
                action_physique = robot.set_lift_height(0.0, in_parallel=True)
            elif action == "avancer":
                dist_mm = val * 10 if val else 50  # cm → mm, défaut 50mm
                action_physique = robot.drive_straight(
                    cozmo.util.distance_mm(dist_mm), cozmo.util.speed_mmps(50),
                    in_parallel=True)
            elif action == "reculer":
                dist_mm = val * 10 if val else 50
                action_physique = robot.drive_straight(
                    cozmo.util.distance_mm(-dist_mm), cozmo.util.speed_mmps(50),
                    in_parallel=True)
            elif action == "tourner_gauche":
                deg = val if val else 90
                action_physique = robot.turn_in_place(
                    cozmo.util.degrees(deg), in_parallel=True)
            elif action == "tourner_droite":
                deg = val if val else 90
                action_physique = robot.turn_in_place(
                    cozmo.util.degrees(-deg), in_parallel=True)

            # --- Widget Météo OLED ---
            action_ecran = None
            if ville:
                try:
                    temp_c, condition_meteo = await loop.run_in_executor(
                        None, obtenir_meteo_sync, ville
                    )
                    if temp_c is not None and condition_meteo is not None:
                        image_oled = generer_image_meteo(
                            ville, temp_c, condition_meteo)
                        screen_data = cozmo.oled_face.convert_image_to_screen_data(
                            image_oled)
                        action_ecran = robot.display_oled_face_image(
                            screen_data, 5000.0, in_parallel=True)
                except Exception:
                    pass

            # --- Exécution parallèle ---
            taches = [
                action_parole.wait_for_completed(),
                action_anim.wait_for_completed()
            ]
            if action_physique is not None:
                taches.append(action_physique.wait_for_completed())
            if action_ecran is not None:
                taches.append(action_ecran.wait_for_completed())

            await asyncio.gather(*taches)

        except Exception:
            robot.abort_all_actions()
            await robot.say_text(
                "Erreur système.", in_parallel=True
            ).wait_for_completed()

        finally:
            robot.set_all_backpack_lights(cozmo.lights.green_light)


if __name__ == "__main__":
    cozmo.run_program(cozmo_program)