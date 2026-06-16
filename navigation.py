"""
navigation.py — Waypoint Navigation "Stop, Look, Think, Act"
=============================================================
Architecture séquentielle avec surveillance anti-chute concurrente :
  Stop → Photo → Odométrie → Gemini (Chain of Thought) → Act‖Cliff → Loop

Sécurité : capteur IR cliff surveillé en continu (20 Hz) pendant chaque
mouvement via asyncio.wait(FIRST_COMPLETED). Zéro OpenCV.
Tout le raisonnement visuel est délégué à Gemini.

Compatible Python 3.8 strict (pas d'asyncio.to_thread).
"""

import cozmo
import os
import json
import requests
import asyncio
import functools
import base64
import io
import time
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

# =====================================================================
# --- CONSTANTES DE NAVIGATION ---
# =====================================================================
DUREE_MAX_NAVIGATION = 300   # secondes avant abandon (5 minutes)
MAX_HISTORIQUE_NAV = 6       # 3 tours user/model max dans la mémoire FIFO


# =====================================================================
# --- CAPTURE SNAPSHOT ---
# =====================================================================
def capturer_snapshot_b64(robot):
    """Capture l'image courante et retourne le JPEG en base64."""
    latest = robot.world.latest_image
    if latest is None or latest.raw_image is None:
        return None

    pil_img = latest.raw_image.resize((320, 240))
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=70)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# =====================================================================
# --- SURVEILLANCE ANTI-CHUTE (Capteur IR cliff — 20 Hz) ---
# =====================================================================
async def surveiller_falaise(robot):
    """Boucle de surveillance du capteur infrarouge de vide.
    Tourne en parallèle d'un mouvement physique.
    Retourne True dès qu'un vide est détecté.
    """
    while True:
        if robot.is_cliff_detected:
            return True
        await asyncio.sleep(0.05)  # Check à 20 Hz


# =====================================================================
# --- CERVEAU NAVIGATEUR (Chain of Thought — Gemini Cloud) ---
# =====================================================================
async def appel_gemini_navigation_async(image_b64, commande_complete, derniere_action,
                                  odometrie, historique_nav):
    """
    Appel REST asynchrone via aiohttp.
    Intègre la commande complète de l'utilisateur, l'odométrie,
    la dernière action, et l'historique FIFO.
    Retourne le dict JSON Chain of Thought ou None en cas d'erreur.
    """
    import aiohttp
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-3.1-flash-lite:generateContent?key={API_KEY}"
    )

    system_prompt = (
        "TU ES LE NAVIGATEUR STRATÉGIQUE D'UN ROBOT COZMO.\n"
        "Tu reçois une image de sa caméra frontale, l'objectif final, "
        "et ton odométrie actuelle (X: avance, Y: latéral, Cap: orientation en degrés).\n\n"
        "Réponds UNIQUEMENT avec ce JSON strict :\n"
        '{\n'
        '  "diagnostic_visuel": "Ce que je vois et comment je me situe '
        'par rapport à la mission (ex: J\'ai avancé de 30cm, je vois la '
        'boîte Nike à 10cm, la tour est derrière).",\n'
        '  "etape_actuelle": "Sous-objectif immédiat '
        '(ex: Manhattan étape B — longer l\'obstacle par la droite).",\n'
        '  "action": "avancer | pivoter_gauche | pivoter_droite | reculer | stop",\n'
        '  "valeur": nombre (distance en cm ou angle en degrés)\n'
        '}\n\n'
        "RÈGLE 1 — LA MARCHE AVANT :\n"
        "Ton but principal est d'AVANCER vers l'objectif. "
        "Si la voie est libre en face, l'action DOIT être 'avancer' "
        "(valeur entre 10 et 30 cm).\n\n"
        "RÈGLE 2 — LE CONTOURNEMENT OBLIGATOIRE (Pattern de Manhattan) :\n"
        "Si un obstacle te bloque la route directe, "
        "suis ce protocole géométrique en 5 étapes :\n"
        "  Étape A : Tourne à 90° (gauche ou droite).\n"
        "  Étape B : Avance pour te décaler latéralement. "
        "LA DISTANCE EST DYNAMIQUE : "
        "si l'obstacle est fin (chaussure, petite boîte), avance de 15 à 20 cm. "
        "Si l'obstacle est très large (meuble, valise, mur de boîtes), "
        "avance d'une plus grande distance (30 à 60 cm) pour être certain "
        "de pouvoir le longer. "
        "Évalue la largeur de l'obstacle sur l'image avant de fixer la valeur. "
        "Attention aux murs de la pièce : garde une petite marge de sécurité.\n"
        "  Étape C : Tourne de 90° dans le SENS INVERSE pour retrouver "
        "ton Cap d'origine (Cap ≈ 0°).\n"
        "  Étape D : Avance de 20 à 30 cm pour dépasser l'obstacle "
        "par le flanc.\n"
        "  Étape E : Tourne doucement (15° à 30°) pour réaligner "
        "ton axe vers l'objectif final.\n"
        "Si tu remarques dans l'image que ton étape de contournement "
        "t'a amené face à un mur de la pièce, ordonne un 'pivoter' "
        "immédiat pour te réorienter, ou un 'reculer' court.\n"
        "Indique toujours dans 'etape_actuelle' à quelle étape Manhattan "
        "tu te trouves (ex: 'Manhattan étape B — longer par la droite').\n\n"
        "RÈGLE 3 — ANTI-PANIQUE :\n"
        "NE JAMAIS enchaîner plus de deux pivots consécutifs sans avancer. "
        "Si ta dernière action ET l'avant-dernière étaient des pivots, "
        "ton action suivante DOIT être 'avancer' (au moins 10 cm) "
        "pour changer ta position (X, Y). "
        "Pivoter sur place sans avancer est INTERDIT au-delà de 2 pivots.\n\n"
        "RÈGLE 4 — ARRÊT STRICT :\n"
        "Ne déclare JAMAIS la mission réussie (action 'stop' sans abandon) "
        "si la base de l'objectif final ne touche pas le VRAI BORD INFÉRIEUR "
        "de ta caméra. Tant que ce n'est pas le cas, tu n'y es pas. "
        "Continue de chercher un chemin.\n\n"
        "VALEURS :\n"
        "- 'avancer'/'reculer' : 10 à 30 cm selon la situation.\n"
        "- 'pivoter_gauche'/'pivoter_droite' : 90° pour le contournement, "
        "15-45° pour les ajustements fins.\n"
        "- La valeur doit TOUJOURS être un nombre positif.\n"
    )

    # Construction du message utilisateur avec contexte intégral + odométrie
    x, y, cap = odometrie
    user_text = (
        f"Mission initiale : {commande_complete}. "
        f"Dernière action exécutée : {derniere_action}. "
        f"Odométrie actuelle : X = {x:.1f} mm, Y = {y:.1f} mm, Cap = {cap:.1f}°. "
        f"Analyse l'image et donne la prochaine étape."
    )

    user_parts = [{"text": user_text}]
    if image_b64:
        user_parts.append({
            "inlineData": {"mimeType": "image/jpeg", "data": image_b64}
        })

    # Construction du payload avec historique FIFO
    contents = list(historique_nav) + [{"role": "user", "parts": user_parts}]

    payload = {
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": contents
    }

    # --- Retry avec backoff exponentiel (429 Too Many Requests) ---
    max_tentatives = 3
    delai = 2

    async with aiohttp.ClientSession() as session:
        for tentative in range(1, max_tentatives + 1):
            try:
                async with session.post(url, headers={"Content-Type": "application/json"}, json=payload, timeout=15) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                    raw = data["candidates"][0]["content"]["parts"][0]["text"]
                    raw = raw.replace("```json", "").replace("```", "").strip()
                    return json.loads(raw)
            except aiohttp.ClientResponseError as e:
                if e.status == 429 and tentative < max_tentatives:
                    await asyncio.sleep(delai)
                    delai *= 2
                else:
                    print(f"⚠️ Erreur API : {e}")
                    return None
            except Exception as e:
                print(f"⚠️ Erreur API : {e}")
                return None

    return None


# =====================================================================
# --- ORCHESTRATEUR "STOP, LOOK, THINK, ACT" ---
# =====================================================================
async def lancer_exploration(robot, commande_complete, broadcast_data=None):
    """
    Boucle Waypoint Navigation purement séquentielle :
      1. STOP  — Le robot est à l'arrêt total
      2. LOOK  — Capture image + Odométrie
      3. THINK — Appel Gemini (Chain of Thought)
      4. ACT   — Mouvement synchrone (attend la fin)
      5. LOOP  — Retour à l'étape 1

    Sécurité anti-chute : check robot.is_cliff_detected avant chaque mouvement.
    Si déclenché → recul d'urgence + injection de contexte pour Gemini.

    Paramètres :
        robot : instance cozmo.robot.Robot
        commande_complete : la commande COMPLÈTE de l'utilisateur
    """
    loop = asyncio.get_event_loop()

    # Préparation physique : tête inclinée vers le bas pour cadrer le sol
    await robot.set_head_angle(cozmo.util.degrees(-15)).wait_for_completed()
    await robot.set_lift_height(1.0).wait_for_completed()

    # État de navigation
    historique_nav = []
    derniere_action = "Démarrage"
    compteur_vide = 0
    debut = asyncio.get_event_loop().time()
    session_id = time.strftime("Exploration %H:%M:%S")

    while True:
        # --- Vérification du timeout ---
        elapsed = asyncio.get_event_loop().time() - debut
        if elapsed >= DUREE_MAX_NAVIGATION:
            print("⚠️ ALERTE : Timeout atteint. Fin de l'exploration.")
            break

        # --- 1. STOP : Le robot est à l'arrêt total ---
        robot.stop_all_motors()

        # --- 2. LOOK : Capture image ---
        await asyncio.sleep(0.3)  # Stabilisation caméra après mouvement

        try:
            img_b64 = await loop.run_in_executor(
                None, functools.partial(capturer_snapshot_b64, robot)
            )
        except Exception:
            img_b64 = None

        if img_b64 is None:
            await asyncio.sleep(1.0)
            continue

        # --- 2b. LOOK : Extraction de l'odométrie ---
        x = robot.pose.position.x
        y = robot.pose.position.y
        cap = robot.pose.rotation.angle_z.degrees
        odometrie = (x, y, cap)

        if broadcast_data:
            await broadcast_data({
                "type": "state_update",
                "session_id": session_id,
                "state": "LOOK",
                "odometry": {"x": x, "y": y, "cap": cap},
                "cliff_detected": robot.is_cliff_detected
            })

        # --- 3. THINK : Appel Gemini (Chain of Thought) ---
        print("🧠 Réfléchis...")
        if broadcast_data:
            await broadcast_data({"type": "state_update", "state": "THINK", "session_id": session_id})

        decision = await appel_gemini_navigation_async(
            img_b64, commande_complete, derniere_action, odometrie, historique_nav
        )

        if decision is None:
            print("⚠️ Erreur API : pas de réponse. Nouvelle tentative dans 4s...")
            await asyncio.sleep(4.0)
            continue

        # --- Extraction des champs Chain of Thought ---
        diagnostic = decision.get("diagnostic_visuel", "?")
        etape = decision.get("etape_actuelle", "?")
        action = decision.get("action", "avancer")
        valeur = decision.get("valeur", 10)

        # Sécurité : s'assurer que la valeur est un nombre positif
        try:
            valeur = abs(float(valeur))
            if valeur == 0:
                valeur = 10
        except (TypeError, ValueError):
            valeur = 10

        if broadcast_data:
            await broadcast_data({
                "type": "ai_cot",
                "session_id": session_id,
                "etape": etape,
                "diagnostic": diagnostic,
                "action": action,
                "valeur": valeur,
                "raw": decision,
                "odometry": {
                    "x": x,
                    "y": y,
                    "cap": cap
                },
                "frame": img_b64
            })

        print(f"🎯 Étape : {etape}")

        # --- Mise à jour de l'historique FIFO ---
        historique_nav.append({
            "role": "user",
            "parts": [{"text":
                f"Mission initiale : {commande_complete}. "
                f"Dernière action exécutée : {derniere_action}. "
                f"Odométrie : X={x:.1f}, Y={y:.1f}, Cap={cap:.1f}°."
            }]
        })
        historique_nav.append({
            "role": "model",
            "parts": [{"text": json.dumps(decision, ensure_ascii=False)}]
        })

        # Tronquer l'historique FIFO
        while len(historique_nav) > MAX_HISTORIQUE_NAV:
            historique_nav.pop(0)

        # --- 4. ACT : Vérifier si stop ---
        if action == "stop":
            robot.stop_all_motors()
            if "abandon" in etape.lower():
                print(f"❌ Mission annulée : {etape}")
                await robot.play_anim_trigger(
                    cozmo.anim.Triggers.CodeLabDejected
                ).wait_for_completed()
            else:
                print("✅ Cible atteinte.")
                await robot.play_anim_trigger(
                    cozmo.anim.Triggers.CodeLabHappy
                ).wait_for_completed()
            break

        # --- 4. ACT : Préparer l'action physique ---
        mouvement = None

        if action == "avancer":
            print(f"🏎️ Action : avancer de {valeur} cm")
            mouvement = robot.drive_straight(
                cozmo.util.distance_mm(valeur * 10),
                cozmo.util.speed_mmps(40)
            )

        elif action == "reculer":
            print(f"🏎️ Action : reculer de {valeur} cm")
            mouvement = robot.drive_straight(
                cozmo.util.distance_mm(-valeur * 10),
                cozmo.util.speed_mmps(40)
            )

        elif action == "pivoter_gauche":
            print(f"🏎️ Action : pivoter_gauche de {valeur}°")
            mouvement = robot.turn_in_place(
                cozmo.util.degrees(valeur)
            )

        elif action == "pivoter_droite":
            print(f"🏎️ Action : pivoter_droite de {valeur}°")
            mouvement = robot.turn_in_place(
                cozmo.util.degrees(-valeur)
            )

        if mouvement is None:
            derniere_action = f"{action} de {valeur}"
            continue

        if broadcast_data:
            await broadcast_data({"type": "state_update", "state": "ACT", "session_id": session_id})

        # --- 4b. ACT : Exécution avec surveillance anti-chute concurrente ---
        tache_mouvement = asyncio.ensure_future(mouvement.wait_for_completed())
        tache_falaise = asyncio.ensure_future(surveiller_falaise(robot))

        done, pending = await asyncio.wait(
            {tache_mouvement, tache_falaise},
            return_when=asyncio.FIRST_COMPLETED
        )

        # Annuler les tâches restantes
        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        # --- Cas nominal : le mouvement s'est terminé normalement ---
        if tache_mouvement in done:
            derniere_action = f"{action} de {valeur}"

        # --- Cas d'urgence : vide détecté pendant le mouvement ---
        else:
            robot.abort_all_actions()
            compteur_vide += 1
            print(f"⚠️ ALERTE : Vide détecté ! Recul d'urgence. ({compteur_vide}/5)")
            if broadcast_data:
                await broadcast_data({"type": "state_update", "cliff_detected": True, "session_id": session_id})

            # --- Abandon matériel : seuil de 5 précipices atteint ---
            if compteur_vide >= 5:
                print(
                    "❌ Mission annulée : ABANDON MATÉRIEL. "
                    "Le système a détecté 5 précipices. "
                    "Le robot est sur une île ou une table."
                )
                await robot.play_anim_trigger(
                    cozmo.anim.Triggers.CodeLabDejected
                ).wait_for_completed()
                break

            # --- Micro-recul de sécurité ---
            await robot.drive_straight(
                cozmo.util.distance_mm(-30),
                cozmo.util.speed_mmps(50)
            ).wait_for_completed()

            # Injection de contexte pour Gemini au prochain cycle
            derniere_action = (
                "INTERRUPTION (vide détecté par capteur infrarouge). "
                "Système de sécurité : Vide détecté devant moi ! "
                "J'ai reculé de 3 cm. URGENCE : N'applique PAS le pattern "
                "Manhattan. Pivote immédiatement de 90° ou 180° pour "
                "changer radicalement de cap, puis avance."
            )
            historique_nav.append({
                "role": "user",
                "parts": [{"text":
                    "Système de sécurité : Vide détecté devant moi ! "
                    "J'ai reculé de 3 cm. URGENCE : N'applique PAS le pattern "
                    "Manhattan. Pivote immédiatement de 90° ou 180° pour "
                    "changer radicalement de cap, puis avance."
                }]
            })
            historique_nav.append({
                "role": "model",
                "parts": [{"text": json.dumps({
                    "diagnostic_visuel": "Vide détecté par capteur infrarouge",
                    "etape_actuelle": "Changement radical de cap après précipice",
                    "action": "reculer",
                    "valeur": 3
                }, ensure_ascii=False)}]
            })
            while len(historique_nav) > MAX_HISTORIQUE_NAV:
                historique_nav.pop(0)

    # --- Fin de l'exploration ---
    robot.stop_all_motors()
    await robot.set_head_angle(cozmo.util.degrees(30)).wait_for_completed()
    robot.set_all_backpack_lights(cozmo.lights.green_light)
