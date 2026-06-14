"""
hybrid_navigation.py — Phase 4 : Navigation Hybride NN + Gemini
================================================================
Deux boucles imbriquées :
  • Boucle LENTE (Gemini, ~5s)  : stratégie de haut niveau, cap cible
  • Boucle RAPIDE (NN, 20 Hz)   : pilotage continu des roues

Compatible Python 3.8 strict.
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
import math
import numpy as np
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

# Import conditionnel PyTorch
TORCH_AVAILABLE = False
try:
    import torch
    from model import CozmoNN, CozmoNNv2
    TORCH_AVAILABLE = True
except ImportError:
    pass

# =====================================================================
# --- CONFIGURATION HYBRIDE ---
# =====================================================================
HYBRID_CONFIG = {
    "use_nn_heading_input": False,   # False=Option B (PID), True=Option A (13 inputs)
    "kp_correction": 1.0,            # Gain proportionnel pour Option B
    "gemini_interval_s": 5.0,        # Intervalle normal Gemini
    "gemini_fast_interval_s": 2.0,   # Si changement cap > 45°
    "cap_change_threshold": 45.0,    # Seuil (degrés) pour accélérer Gemini
    "timeout_s": 300.0,              # 5 minutes max
}


# =====================================================================
# --- UTILITAIRES ---
# =====================================================================
def normalize_angle(angle):
    """Normalise un angle en degrés dans [-180, 180]."""
    while angle > 180:
        angle -= 360
    while angle < -180:
        angle += 360
    return angle


def find_latest_model():
    """Trouve le modèle le plus récent dans le dossier models/."""
    models_dir = "models"
    if not os.path.exists(models_dir):
        return None
    pt_files = [f for f in os.listdir(models_dir) if f.endswith(".pt")]
    if not pt_files:
        return None
    # Sort by modification time, newest first
    pt_files.sort(key=lambda f: os.path.getmtime(os.path.join(models_dir, f)), reverse=True)
    return pt_files[0][:-3]  # Return name without .pt


# =====================================================================
# --- VECTEUR SENSORIEL (copie de main.py pour indépendance du module)
# =====================================================================
def get_sensor_vector(robot):
    """Retourne un vecteur de 12 capteurs pour le modèle NN."""
    sensor_vector = [0.0] * 12
    if robot is None:
        return sensor_vector

    # 1. robot.pose.position.x
    try:
        sensor_vector[0] = float(robot.pose.position.x)
    except Exception:
        pass
    # 2. robot.pose.position.y
    try:
        sensor_vector[1] = float(robot.pose.position.y)
    except Exception:
        pass
    # 3. robot.pose.rotation.angle_z.degrees
    try:
        sensor_vector[2] = float(robot.pose.rotation.angle_z.degrees)
    except Exception:
        pass
    # 4. robot.pose_pitch.degrees
    try:
        sensor_vector[3] = float(robot.pose_pitch.degrees)
    except Exception:
        pass
    # 5. robot.left_wheel_speed.speed_mmps
    try:
        if robot.left_wheel_speed is not None:
            sensor_vector[4] = float(robot.left_wheel_speed.speed_mmps)
    except Exception:
        pass
    # 6. robot.right_wheel_speed.speed_mmps
    try:
        if robot.right_wheel_speed is not None:
            sensor_vector[5] = float(robot.right_wheel_speed.speed_mmps)
    except Exception:
        pass
    # 7. 1.0 if robot.is_cliff_detected else 0.0
    try:
        sensor_vector[6] = 1.0 if robot.is_cliff_detected else 0.0
    except Exception:
        pass
    # 8. robot.lift_height.distance_mm
    try:
        if robot.lift_height is not None:
            sensor_vector[7] = float(robot.lift_height.distance_mm)
    except Exception:
        pass
    # 9. robot.head_angle.degrees
    try:
        if robot.head_angle is not None:
            sensor_vector[8] = float(robot.head_angle.degrees)
    except Exception:
        pass
    # 10. robot.battery_voltage
    try:
        if robot.battery_voltage is not None:
            sensor_vector[9] = float(robot.battery_voltage)
    except Exception:
        pass
    # 11. robot.pose.rotation.q0
    try:
        if robot.pose is not None and robot.pose.rotation is not None:
            sensor_vector[10] = float(robot.pose.rotation.q0)
    except Exception:
        pass
    # 12. 1.0 if robot.is_moving else 0.0
    try:
        sensor_vector[11] = 1.0 if robot.is_moving else 0.0
    except Exception:
        pass
    return sensor_vector


# =====================================================================
# --- CHARGEMENT MODÈLE + STATS ---
# =====================================================================
def load_hybrid_model_sync(model_name, use_v2=False):
    """
    Charge un modèle NN et ses stats de normalisation (synchrone).
    Retourne (model, mean, std) ou (None, None, None) en cas d'erreur.
    """
    try:
        # Strip .pt if present
        model_key = model_name
        if model_key.endswith(".pt"):
            model_key = model_key[:-3]

        if model_key.startswith("cozmo_nn_"):
            suffix = model_key[len("cozmo_nn_"):]
        else:
            suffix = model_key

        model_path = os.path.join("models", "{}.pt".format(model_key))
        norm_stats_path = os.path.join("models", "norm_stats_{}.json".format(suffix))

        if not os.path.exists(model_path):
            print("Model file not found: {}".format(model_path))
            return None, None, None

        if not os.path.exists(norm_stats_path):
            print("Norm stats file not found: {}".format(norm_stats_path))
            return None, None, None

        # Load stats
        with open(norm_stats_path, 'r') as f:
            stats = json.load(f)
        mean = np.array(stats['mean'], dtype=np.float32)
        std = np.array(stats['std'], dtype=np.float32)

        # Instantiate model
        if use_v2:
            model = CozmoNNv2()
        else:
            model = CozmoNN()
        state_dict = torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()

        return model, mean, std
    except Exception as e:
        print("Error loading hybrid model: {}".format(e))
        return None, None, None


# =====================================================================
# --- APPEL GEMINI STRATÉGIQUE ---
# =====================================================================
def appel_gemini_strategique_sync(image_b64, commande_utilisateur, odometrie,
                                  historique_strategique):
    """
    Appel REST synchrone au modèle Gemini pour la stratégie de navigation.
    Exécuté dans un thread via run_in_executor.
    Retourne le dict JSON stratégique ou None.
    """
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        "gemini-3.1-flash-lite:generateContent?key={}".format(API_KEY)
    )

    system_prompt = (
        "TU ES LE NAVIGATEUR STRATÉGIQUE d'un robot Cozmo. "
        "Un réseau de neurones pilote le robot en continu pour éviter les obstacles. "
        "TON rôle est de donner la DIRECTION STRATÉGIQUE.\n\n"
        "Tu reçois : une image caméra + l'odométrie + l'objectif de l'utilisateur.\n\n"
        "Réponds UNIQUEMENT avec ce JSON :\n"
        "{\n"
        '  "cap_cible_degres": nombre (le cap absolu en degrés vers lequel le robot doit se diriger),\n'
        '  "description_scene": "Ce que tu vois devant le robot",\n'
        '  "strategie": "Ta stratégie de haut niveau",\n'
        '  "objectif_atteint": false,\n'
        '  "confiance": 0.8\n'
        "}\n\n"
        "RÈGLES :\n"
        "- Le cap_cible est en degrés ABSOLUS par rapport au repère d'origine du robot "
        "(0° = direction de départ)\n"
        "- Ne donne PAS de commandes de mouvement — le NN gère ça\n"
        "- Concentre-toi sur la STRATÉGIE : vers où aller, faut-il contourner, "
        "est-on proche de l'objectif\n"
        "- Si tu vois l'objectif, indique le cap vers lui\n"
        "- Si la voie est bloquée, indique un cap de contournement\n"
        "- objectif_atteint = true UNIQUEMENT si l'objectif est au premier plan de l'image"
    )

    x, y, cap = odometrie
    user_text = (
        "Objectif : {}. "
        "Odométrie actuelle : X = {:.1f} mm, Y = {:.1f} mm, Cap = {:.1f}°. "
        "Analyse l'image et donne le cap cible."
    ).format(commande_utilisateur, x, y, cap)

    user_parts = [{"text": user_text}]
    if image_b64:
        user_parts.append({
            "inlineData": {"mimeType": "image/jpeg", "data": image_b64}
        })

    contents = list(historique_strategique) + [{"role": "user", "parts": user_parts}]
    payload = {
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": contents
    }

    # --- Retry avec backoff exponentiel (429 Too Many Requests) ---
    max_tentatives = 3
    delai = 2

    for tentative in range(1, max_tentatives + 1):
        try:
            resp = requests.post(url, headers={"Content-Type": "application/json"},
                                 json=payload, timeout=15)
            resp.raise_for_status()
            raw = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
            raw = raw.replace("```json", "").replace("```", "").strip()
            return json.loads(raw)
        except requests.exceptions.HTTPError as e:
            if resp.status_code == 429 and tentative < max_tentatives:
                time.sleep(delai)
                delai *= 2
            else:
                print("⚠️ Erreur API stratégique : {}".format(e))
                return None
        except Exception as e:
            print("⚠️ Erreur API stratégique : {}".format(e))
            return None

    return None


# =====================================================================
# --- BOUCLE GEMINI STRATÉGIQUE (~5 secondes) ---
# =====================================================================
async def boucle_gemini_strategique(robot, commande_utilisateur, shared_state,
                                    broadcast_data):
    """
    Boucle lente : capture une image 320x240, lit l'odométrie, appelle
    Gemini via run_in_executor, met à jour shared_state avec le cap cible.
    Adapte son intervalle : 5s normal, 2s si le cap a changé > 45°.
    """
    loop = asyncio.get_event_loop()
    historique_strategique = []
    max_historique = 6
    ancien_cap = 0.0

    while shared_state["gemini_active"] and not shared_state["kill_switch"]:
        # --- Capture image 320x240 pour Gemini ---
        try:
            latest = robot.world.latest_image
            if latest is not None and latest.raw_image is not None:
                def capture_gemini_image(img):
                    pil_img = img.resize((320, 240))
                    buf = io.BytesIO()
                    pil_img.save(buf, format="JPEG", quality=75)
                    return base64.b64encode(buf.getvalue()).decode("utf-8")
                image_b64 = await loop.run_in_executor(
                    None, capture_gemini_image, latest.raw_image
                )
            else:
                image_b64 = None
        except Exception:
            image_b64 = None

        if image_b64 is None:
            await asyncio.sleep(1.0)
            continue

        # --- Lecture odométrie ---
        x = robot.pose.position.x
        y = robot.pose.position.y
        cap = robot.pose.rotation.angle_z.degrees
        odometrie = (x, y, cap)

        # --- Appel Gemini dans un thread ---
        try:
            decision = await loop.run_in_executor(
                None,
                functools.partial(
                    appel_gemini_strategique_sync,
                    image_b64, commande_utilisateur, odometrie, historique_strategique
                )
            )
        except Exception as e:
            print("⚠️ Erreur boucle Gemini : {}".format(e))
            decision = None

        if decision is not None:
            # --- Extraction des champs ---
            nouveau_cap = float(decision.get("cap_cible_degres", shared_state["cap_cible"]))
            description = decision.get("description_scene", "")
            strategie = decision.get("strategie", "")
            objectif = decision.get("objectif_atteint", False)
            confiance = float(decision.get("confiance", 0.0))

            # --- Mise à jour shared_state ---
            shared_state["cap_cible"] = nouveau_cap
            shared_state["description_scene"] = description
            shared_state["strategie"] = strategie
            shared_state["objectif_atteint"] = objectif
            shared_state["confiance"] = confiance

            # --- Mise à jour historique FIFO ---
            historique_strategique.append({
                "role": "user",
                "parts": [{"text":
                    "Objectif : {}. Odométrie : X={:.1f}, Y={:.1f}, Cap={:.1f}°.".format(
                        commande_utilisateur, x, y, cap
                    )
                }]
            })
            historique_strategique.append({
                "role": "model",
                "parts": [{"text": json.dumps(decision, ensure_ascii=False)}]
            })
            while len(historique_strategique) > max_historique:
                historique_strategique.pop(0)

            # --- Broadcast vers le frontend ---
            if broadcast_data:
                await broadcast_data({
                    "type": "hybrid_gemini",
                    "cap_cible": nouveau_cap,
                    "description_scene": description,
                    "strategie": strategie,
                    "confiance": confiance,
                    "objectif_atteint": objectif,
                    "odometry": {"x": x, "y": y, "cap": cap}
                })

            print("🧭 Gemini cap={:.1f}° | {} | conf={:.1f}".format(
                nouveau_cap, strategie, confiance
            ))

            # --- Vérification objectif atteint ---
            if objectif:
                shared_state["gemini_active"] = False
                if broadcast_data:
                    await broadcast_data({
                        "type": "terminal_log",
                        "message": "✅ Gemini : objectif atteint !",
                        "source": "system"
                    })
                break

            # --- Adapter l'intervalle selon le changement de cap ---
            delta_cap = abs(normalize_angle(nouveau_cap - ancien_cap))
            ancien_cap = nouveau_cap

            if delta_cap > HYBRID_CONFIG["cap_change_threshold"]:
                intervalle = HYBRID_CONFIG["gemini_fast_interval_s"]
            else:
                intervalle = HYBRID_CONFIG["gemini_interval_s"]
        else:
            # Erreur Gemini : attendre un peu plus
            intervalle = HYBRID_CONFIG["gemini_interval_s"]

        # --- Attente avant le prochain cycle ---
        await asyncio.sleep(intervalle)


# =====================================================================
# --- BOUCLE NN RÉACTIVE (20 Hz) ---
# =====================================================================
async def boucle_nn_reactive(robot, nn_model, sensors_mean, sensors_std,
                             shared_state, broadcast_data):
    """
    Boucle rapide à 20Hz : capture image 80x60 grayscale, normalise les
    capteurs, exécute le NN, applique la correction de cap, pilote les roues.
    Ne bloque JAMAIS en attente de Gemini.
    """
    loop = asyncio.get_event_loop()

    fps_ema = 20.0
    inf_ms_ema = 0.0
    last_tick_time = 0.0
    tick_count = 0
    cliff_events = 0

    use_v2 = HYBRID_CONFIG["use_nn_heading_input"]

    while shared_state["nn_active"] and not shared_state["kill_switch"]:
        start_time = loop.time()

        # --- Calcul FPS ---
        t_now = loop.time()
        if last_tick_time > 0:
            dt = t_now - last_tick_time
            current_fps = 1.0 / dt if dt > 0 else 20.0
            fps_ema = 0.9 * fps_ema + 0.1 * current_fps
        last_tick_time = t_now

        # --- Sécurité anti-chute (cliff override) ---
        if robot.is_cliff_detected:
            cliff_events += 1
            await robot.stop_all_motors()
            await robot.drive_straight(
                cozmo.util.distance_mm(-50),
                cozmo.util.speed_mmps(50)
            ).wait_for_completed()
            await asyncio.sleep(0.5)
            continue

        # --- 1. Capture image (80x60 grayscale, normalisée [0,1]) ---
        try:
            latest = robot.world.latest_image
            if latest is not None and latest.raw_image is not None:
                def process_image(img):
                    gray_img = img.convert('L').resize((80, 60))
                    return np.array(gray_img, dtype=np.float32) / 255.0
                img_arr = await loop.run_in_executor(None, process_image, latest.raw_image)
            else:
                img_arr = np.zeros((60, 80), dtype=np.float32)
        except Exception:
            img_arr = np.zeros((60, 80), dtype=np.float32)

        # --- 2. Capture vecteur sensoriel ---
        sensors = get_sensor_vector(robot)

        # --- 3. Normaliser les capteurs ---
        try:
            sensors_normalized = (np.array(sensors, dtype=np.float32) - sensors_mean[0]) / sensors_std[0]
        except Exception:
            sensors_normalized = np.zeros((12,), dtype=np.float32)

        # --- 4. Lire cap_cible et calculer delta_heading ---
        cap_actuel = robot.pose.rotation.angle_z.degrees
        delta_heading_deg = normalize_angle(shared_state["cap_cible"] - cap_actuel)
        delta_heading_normalized = max(-1.0, min(1.0, delta_heading_deg / 180.0))

        # --- 5. Inférence NN ---
        t_inf_start = time.time()
        try:
            img_tensor = torch.from_numpy(img_arr).unsqueeze(0).unsqueeze(0)  # (1, 1, 60, 80)

            if use_v2:
                # Option A : 13 inputs (delta_heading comme 13ème entrée)
                sensor_with_heading = np.append(sensors_normalized, delta_heading_normalized)
                sensor_tensor = torch.from_numpy(sensor_with_heading).unsqueeze(0)  # (1, 13)
                with torch.no_grad():
                    outputs = nn_model(img_tensor, sensor_tensor)
                left_speed = float(outputs[0, 0].item())
                right_speed = float(outputs[0, 1].item())
                head_angle = float(outputs[0, 2].item())
            else:
                # Option B : 12 inputs + correction PID sur le cap
                sensor_tensor = torch.from_numpy(sensors_normalized).unsqueeze(0)  # (1, 12)
                with torch.no_grad():
                    outputs = nn_model(img_tensor, sensor_tensor)
                nn_left = float(outputs[0, 0].item())
                nn_right = float(outputs[0, 1].item())
                head_angle = float(outputs[0, 2].item())

                # Correction proportionnelle du cap
                kp = HYBRID_CONFIG["kp_correction"]
                correction = kp * delta_heading_normalized * 50.0
                left_speed = nn_left + correction
                right_speed = nn_right - correction

        except Exception as e:
            print("Inference error: {}".format(e))
            left_speed = 0.0
            right_speed = 0.0
            head_angle = robot.head_angle.degrees if robot.head_angle else 0.0

        t_inf_end = time.time()
        inference_ms = (t_inf_end - t_inf_start) * 1000.0
        inf_ms_ema = 0.9 * inf_ms_ema + 0.1 * inference_ms

        # --- 6. Clip et appliquer les commandes aux roues ---
        left_speed = max(-150.0, min(150.0, left_speed))
        right_speed = max(-150.0, min(150.0, right_speed))
        try:
            await robot.drive_wheels(left_speed, right_speed)
        except Exception as e:
            print("Error driving wheels: {}".format(e))

        # --- Ajuster l'angle de la tête ---
        try:
            current_head = robot.head_angle.degrees if robot.head_angle else 0.0
            if abs(head_angle - current_head) > 2.0:
                robot.set_head_angle(cozmo.util.degrees(head_angle), in_parallel=True)
        except Exception:
            pass

        # --- 7. Broadcast vers le dashboard (~4Hz, tous les 5 ticks) ---
        tick_count += 1
        if tick_count % 5 == 0:
            if broadcast_data:
                await broadcast_data({
                    "type": "hybrid_status",
                    "nn_active": True,
                    "gemini_active": shared_state["gemini_active"],
                    "cap_cible": round(shared_state["cap_cible"], 1),
                    "cap_actuel": round(cap_actuel, 1),
                    "delta_heading": round(delta_heading_deg, 1),
                    "left_speed": round(left_speed, 1),
                    "right_speed": round(right_speed, 1),
                    "fps": round(fps_ema, 1),
                    "inference_ms": round(inf_ms_ema, 1),
                    "head_angle": round(head_angle, 1),
                    "cliff_events": cliff_events,
                    "description_scene": shared_state["description_scene"],
                    "strategie": shared_state["strategie"],
                    "confiance": shared_state["confiance"],
                    "objectif_atteint": shared_state["objectif_atteint"]
                })

        # --- Sleep pour maintenir 20Hz ---
        elapsed = loop.time() - start_time
        sleep_time = max(0.001, 0.05 - elapsed)
        await asyncio.sleep(sleep_time)


# =====================================================================
# --- ORCHESTRATEUR HYBRIDE ---
# =====================================================================
async def lancer_exploration_hybride(robot, commande_utilisateur, broadcast_data,
                                     model_name=None, shared_state=None):
    """
    Point d'entrée de l'exploration hybride NN + Gemini.
      1. Trouve et charge le modèle NN
      2. Initialise le shared_state
      3. Configure la tête pour l'exploration (-15°)
      4. Lance les deux boucles en parallèle (create_task)
      5. Attend la terminaison (objectif_atteint, timeout, kill_switch)
      6. Arrêt propre (cancel tasks, stop_all_motors)
    """
    if not TORCH_AVAILABLE:
        if broadcast_data:
            await broadcast_data({
                "type": "terminal_log",
                "message": "❌ Erreur : PyTorch n'est pas disponible.",
                "source": "system"
            })
        return

    loop = asyncio.get_event_loop()

    # --- 1. Trouver et charger le modèle ---
    if model_name is None:
        model_name = find_latest_model()
    if model_name is None:
        if broadcast_data:
            await broadcast_data({
                "type": "terminal_log",
                "message": "❌ Aucun modèle trouvé dans models/",
                "source": "system"
            })
        return

    use_v2 = HYBRID_CONFIG["use_nn_heading_input"]

    if broadcast_data:
        await broadcast_data({
            "type": "terminal_log",
            "message": "🧠 Chargement du modèle {} pour navigation hybride...".format(model_name),
            "source": "system"
        })

    nn_model, sensors_mean, sensors_std = await loop.run_in_executor(
        None,
        functools.partial(load_hybrid_model_sync, model_name, use_v2)
    )

    if nn_model is None:
        if broadcast_data:
            await broadcast_data({
                "type": "terminal_log",
                "message": "❌ Échec du chargement du modèle {}".format(model_name),
                "source": "system"
            })
        return

    if broadcast_data:
        await broadcast_data({
            "type": "terminal_log",
            "message": "✅ Modèle {} chargé. Lancement exploration hybride...".format(model_name),
            "source": "system"
        })

    # --- 2. Initialiser le shared_state (ou utiliser celui passé par main.py) ---
    if shared_state is None:
        shared_state = {
            "cap_cible": 0.0,
            "description_scene": "",
            "strategie": "",
            "objectif_atteint": False,
            "confiance": 0.0,
            "kill_switch": False,
            "nn_active": True,
            "gemini_active": True,
        }
    else:
        # S'assurer que les flags de démarrage sont corrects
        shared_state["nn_active"] = True
        shared_state["gemini_active"] = True
        shared_state["kill_switch"] = False
        shared_state["objectif_atteint"] = False

    # --- 3. Préparer le robot : tête vers le bas pour cadrer le sol ---
    await robot.set_head_angle(cozmo.util.degrees(-15)).wait_for_completed()
    await robot.set_lift_height(1.0).wait_for_completed()

    # --- 4. Lancer les deux boucles en parallèle ---
    tache_gemini = asyncio.create_task(
        boucle_gemini_strategique(robot, commande_utilisateur, shared_state, broadcast_data)
    )
    tache_nn = asyncio.create_task(
        boucle_nn_reactive(robot, nn_model, sensors_mean, sensors_std,
                           shared_state, broadcast_data)
    )

    if broadcast_data:
        await broadcast_data({
            "type": "terminal_log",
            "message": "🚀 Navigation hybride démarrée | NN@20Hz + Gemini@{:.0f}s | Objectif : {}".format(
                HYBRID_CONFIG["gemini_interval_s"], commande_utilisateur
            ),
            "source": "system"
        })

    # --- 5. Attendre la terminaison ---
    debut = loop.time()
    timeout = HYBRID_CONFIG["timeout_s"]

    while True:
        await asyncio.sleep(0.5)

        # Vérifier objectif atteint
        if shared_state["objectif_atteint"]:
            if broadcast_data:
                await broadcast_data({
                    "type": "terminal_log",
                    "message": "🎯 Objectif atteint ! Arrêt de la navigation hybride.",
                    "source": "system"
                })
            break

        # Vérifier kill_switch
        if shared_state["kill_switch"]:
            if broadcast_data:
                await broadcast_data({
                    "type": "terminal_log",
                    "message": "⏹ Navigation hybride arrêtée (kill_switch).",
                    "source": "system"
                })
            break

        # Vérifier timeout
        elapsed = loop.time() - debut
        if elapsed >= timeout:
            if broadcast_data:
                await broadcast_data({
                    "type": "terminal_log",
                    "message": "⏱ Timeout ({:.0f}s). Arrêt de la navigation hybride.".format(timeout),
                    "source": "system"
                })
            break

        # Vérifier si les deux boucles sont mortes
        if tache_gemini.done() and tache_nn.done():
            break

    # --- 6. Arrêt propre ---
    shared_state["nn_active"] = False
    shared_state["gemini_active"] = False
    shared_state["kill_switch"] = True

    # Annuler les tâches restantes
    for tache in [tache_gemini, tache_nn]:
        if not tache.done():
            tache.cancel()
            try:
                await tache
            except asyncio.CancelledError:
                pass

    # Arrêter les moteurs
    try:
        await robot.stop_all_motors()
    except Exception:
        pass

    # Remettre la tête en position normale
    try:
        await robot.set_head_angle(cozmo.util.degrees(30)).wait_for_completed()
    except Exception:
        pass

    # Animation de fin
    try:
        if shared_state["objectif_atteint"]:
            await robot.play_anim_trigger(
                cozmo.anim.Triggers.CodeLabHappy
            ).wait_for_completed()
        else:
            await robot.play_anim_trigger(
                cozmo.anim.Triggers.CodeLabDejected
            ).wait_for_completed()
    except Exception:
        pass

    robot.set_all_backpack_lights(cozmo.lights.green_light)

    if broadcast_data:
        await broadcast_data({
            "type": "hybrid_status",
            "nn_active": False,
            "gemini_active": False,
            "cap_cible": 0.0,
            "cap_actuel": 0.0,
            "delta_heading": 0.0,
            "left_speed": 0.0,
            "right_speed": 0.0,
            "fps": 0.0,
            "inference_ms": 0.0,
            "head_angle": 0.0,
            "cliff_events": 0,
            "description_scene": "",
            "strategie": "",
            "confiance": 0.0,
            "objectif_atteint": shared_state["objectif_atteint"]
        })

    return shared_state
