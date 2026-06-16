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

from nav_benchmark import NavigationRunLogger, distance_from_start_mm, odometry_snapshot

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

# Import conditionnel PyTorch
TORCH_AVAILABLE = False
try:
    import torch
    from model import CozmoNN, CozmoNNv2, CozmoNNDiscrete
    TORCH_AVAILABLE = True
except ImportError:
    pass

# =====================================================================
# --- CONFIGURATION HYBRIDE ---
# =====================================================================
HYBRID_CONFIG = {
    "use_nn_heading_input": False,   # False=Option B (PID), True=Option A (13 inputs)
    "kp_correction": 0.8,            # Gain proportionnel pour Option B
    "gemini_interval_s": 3.0,        # Intervalle normal Gemini
    "gemini_fast_interval_s": 1.0,   # Si changement cap > 45°
    "cap_change_threshold": 45.0,    # Seuil (degrés) pour accélérer Gemini
    "timeout_s": 300.0,              # 5 minutes max
    "obstacle_cap_offset_deg": 36.0,
    "obstacle_speed_scale": 0.45,
    "approach_speed_scale_medium": 0.65,
    "approach_speed_scale_large": 0.35,
    "arrival_confidence": 0.74,
    "arrival_confirmations": 2,
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


def _coerce_bool(value, default=False):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("true", "yes", "oui", "1")
    if value is None:
        return default
    return bool(value)


def _coerce_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_position(value):
    value = str(value or "unknown").strip().lower()
    aliases = {
        "gauche": "left",
        "centre": "center",
        "milieu": "center",
        "droite": "right",
        "aucun": "none",
        "inconnu": "unknown",
    }
    value = aliases.get(value, value)
    if value not in ("left", "center", "right", "unknown", "none"):
        return "unknown"
    return value


def _normalize_side(value):
    value = str(value or "none").strip().lower()
    aliases = {"gauche": "left", "droite": "right", "aucun": "none"}
    value = aliases.get(value, value)
    if value not in ("left", "right", "none"):
        return "none"
    return value


def _normalize_size(value):
    value = str(value or "unknown").strip().lower()
    aliases = {
        "petit": "small",
        "moyen": "medium",
        "grand": "large",
        "tres_grand": "very_large",
        "très grand": "very_large",
        "inconnu": "unknown",
    }
    value = aliases.get(value, value)
    if value not in ("small", "medium", "large", "very_large", "unknown"):
        return "unknown"
    return value


def sanitize_hybrid_decision(raw):
    raw = raw or {}
    action = str(raw.get("action_navigation", "")).strip().upper()
    
    if str(raw.get("target_position")).strip().lower() == "not_visible":
        target_position = "unknown"
        target_visible = False
    else:
        target_position = _normalize_position(raw.get("target_position"))
        target_visible = _coerce_bool(raw.get("target_visible"), action in (
            "CIBLE_CENTREE", "CIBLE_A_DROITE", "CIBLE_A_GAUCHE", "OBJECTIF_ATTEINT"
        ))

    if target_position == "unknown":
        if action == "CIBLE_A_DROITE":
            target_position = "right"
        elif action == "CIBLE_A_GAUCHE":
            target_position = "left"
        elif action in ("CIBLE_CENTREE", "OBJECTIF_ATTEINT"):
            target_position = "center"

    return {
        "cap_cible_degres": _coerce_float(raw.get("cap_cible_degres"), 0.0),
        "target_visible": target_visible,
        "target_position": target_position,
        "target_size": _normalize_size(raw.get("target_size")),
        "target_touching_bottom": _coerce_bool(raw.get("target_touching_bottom"), action == "OBJECTIF_ATTEINT"),
        "obstacle_in_path": _coerce_bool(raw.get("obstacle_in_path")),
        "obstacle_position": _normalize_position(raw.get("obstacle_position")),
        "recommended_side": _normalize_side(raw.get("recommended_side")),
        "arrival_confidence": max(0.0, min(1.0, _coerce_float(raw.get("arrival_confidence"), 0.0))),
        "safe_to_advance": _coerce_bool(raw.get("safe_to_advance"), True),
        "description_scene": str(raw.get("description_scene", "")),
        "strategie": str(raw.get("strategie", "")),
        "reflexion": str(raw.get("reflexion", "")),
        "objectif_atteint": _coerce_bool(raw.get("objectif_atteint")),
        "confiance": max(0.0, min(1.0, _coerce_float(raw.get("confiance"), 0.0))),
        "raw": raw,
    }


def find_latest_model():
    """Trouve le modèle le plus récent dans le dossier models/."""
    models_dir = "models"
    if not os.path.exists(models_dir):
        return None
    pt_files = [
        f for f in os.listdir(models_dir)
        if f.endswith(".pt") and f.startswith("cozmo_nn_")
    ]
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

        model_path = os.path.join("models", "{}.pt".format(model_key))
        norm_stats_path = os.path.join("models", "norm_stats_{}.json".format(model_key))

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
        state_dict = torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)
        
        if "discrete" in model_key:
            if "2_3" in model_key or "2_4" in model_key:
                num_actions = 4
            else:
                num_actions = 7
            model = CozmoNNDiscrete(sensor_dim=8, num_actions=num_actions)
        elif use_v2:
            model = CozmoNNv2()
        else:
            model = CozmoNN()
            
        model.load_state_dict(state_dict)
        model.eval()

        return model, mean, std
    except Exception as e:
        print("Error loading hybrid model: {}".format(e))
        return None, None, None


# =====================================================================
# --- APPEL GEMINI STRATÉGIQUE ---
# =====================================================================
async def appel_gemini_strategique_async(image_b64, commande_utilisateur, odometrie,
                                  historique_strategique):
    """
    Appel REST asynchrone au modèle Gemini via aiohttp.
    Retourne le dict JSON stratégique ou None.
    """
    import aiohttp
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        "gemini-3.1-flash-lite:generateContent?key={}".format(API_KEY)
    )

    system_prompt = (
        "TU ES LA PERCEPTION STRATEGIQUE d'un robot Cozmo. "
        "Tu dois detecter la cible, detecter si un obstacle bloque le chemin direct, "
        "proposer le cote de contournement le plus libre, et estimer si Cozmo est arrive. "
        "Le NN local n'est qu'un reflexe bas niveau: ne suppose jamais qu'il peut resoudre seul "
        "un obstacle visible. Reponds UNIQUEMENT avec ce JSON strict:\n"
        "{\n"
        '  "cap_cible_degres": 0,\n'
        '  "target_visible": true,\n'
        '  "target_position": "center|left|right|not_visible",\n'
        '  "obstacle_in_path": false,\n'
        '  "recommended_side": "left|right|none",\n'
        '  "arrival_confidence": 0.8,\n'
        '  "description_scene": "...",\n'
        '  "strategie": "...",\n'
        '  "objectif_atteint": false,\n'
        '  "confiance": 0.8\n'
        "}\n\n"
        "Regles: cap_cible_degres est le cap global absolu a suivre (utilise ton estimation d'apres l'odometrie actuelle). "
        "obstacle_in_path=true si une boite, un mur ou un objet occupe le chemin direct. "
        "recommended_side est le cote le plus libre pour contourner. "
        "arrival_confidence est eleve si la cible est tres proche. "
        "objectif_atteint=true si l'objectif final est termine."
    )

    x, y, cap = odometrie
    user_text = (
        "Objectif : {}. "
        "Odométrie actuelle : X = {:.1f} mm, Y = {:.1f} mm, Cap = {:.1f}°. "
        "Analyse l'image et donne le JSON de perception strategique."
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

    async with aiohttp.ClientSession() as session:
        for tentative in range(1, max_tentatives + 1):
            try:
                async with session.post(url, headers={"Content-Type": "application/json"}, json=payload, timeout=15) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                    raw = data["candidates"][0]["content"]["parts"][0]["text"]
                    raw = raw.replace("```json", "").replace("```", "").strip()
                    import json
                    return json.loads(raw)
            except aiohttp.ClientResponseError as e:
                if e.status == 429 and tentative < max_tentatives:
                    await asyncio.sleep(delai)
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
                                    broadcast_data, logger=None):
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
            decision = await appel_gemini_strategique_async(
                image_b64, commande_utilisateur, odometrie, historique_strategique
            )
        except Exception as e:
            print("⚠️ Erreur boucle Gemini : {}".format(e))
            decision = None

        if decision is not None:
            clean = sanitize_hybrid_decision(decision)
            description = clean["description_scene"]
            strategie_brute = clean["strategie"]
            confiance = clean["confiance"] if "confiance" in clean["raw"] else clean["arrival_confidence"]

            if clean["objectif_atteint"] or (
                clean["target_visible"]
                and clean["target_touching_bottom"]
                and clean["arrival_confidence"] >= HYBRID_CONFIG["arrival_confidence"]
            ):
                shared_state["arrival_confirmations"] = shared_state.get("arrival_confirmations", 0) + 1
            else:
                shared_state["arrival_confirmations"] = 0

            objectif = (
                shared_state["arrival_confirmations"] >= HYBRID_CONFIG["arrival_confirmations"]
            )

            cap_origine = shared_state.get("cap_origine", 0.0)
            speed_scale = 1.0
            hold_position = False

            if "cap_cible_degres" in clean["raw"]:
                nouveau_cap = normalize_angle(clean["cap_cible_degres"])
                strategie = "[GEMINI_CAP] {}".format(strategie_brute)
                if clean["obstacle_in_path"]:
                    speed_scale = HYBRID_CONFIG["obstacle_speed_scale"]
                elif clean["target_visible"]:
                    if clean["target_size"] in ("large", "very_large"):
                        speed_scale = HYBRID_CONFIG["approach_speed_scale_large"]
                    elif clean["target_size"] == "medium":
                        speed_scale = HYBRID_CONFIG["approach_speed_scale_medium"]
            else:
                if clean["obstacle_in_path"]:
                    side = clean["recommended_side"] if clean["recommended_side"] != "none" else "right"
                    offset = HYBRID_CONFIG["obstacle_cap_offset_deg"]
                    if side == "right":
                        nouveau_cap = normalize_angle(cap - offset)
                    else:
                        nouveau_cap = normalize_angle(cap + offset)
                    speed_scale = HYBRID_CONFIG["obstacle_speed_scale"]
                    strategie = "[OBSTACLE] contournement par {}".format(side)
                elif clean["target_visible"]:
                    if clean["target_position"] == "right":
                        nouveau_cap = normalize_angle(cap - 15.0)
                        strategie = "[TARGET_RIGHT] {}".format(strategie_brute)
                    elif clean["target_position"] == "left":
                        nouveau_cap = normalize_angle(cap + 15.0)
                        strategie = "[TARGET_LEFT] {}".format(strategie_brute)
                    else:
                        nouveau_cap = cap
                        strategie = "[TARGET_CENTER] {}".format(strategie_brute)

                    if clean["target_size"] in ("large", "very_large"):
                        speed_scale = HYBRID_CONFIG["approach_speed_scale_large"]
                    elif clean["target_size"] == "medium":
                        speed_scale = HYBRID_CONFIG["approach_speed_scale_medium"]
                else:
                    nouveau_cap = cap_origine
                    strategie = "[SEARCH] {}".format(strategie_brute)

            if not clean["safe_to_advance"] and not clean["obstacle_in_path"]:
                hold_position = True

            # --- Mise à jour shared_state ---
            shared_state["cap_cible"] = nouveau_cap
            shared_state["description_scene"] = description
            shared_state["strategie"] = strategie
            shared_state["objectif_atteint"] = objectif
            shared_state["confiance"] = confiance
            shared_state["target_visible"] = clean["target_visible"]
            shared_state["target_position"] = clean["target_position"]
            shared_state["target_size"] = clean["target_size"]
            shared_state["target_touching_bottom"] = clean["target_touching_bottom"]
            shared_state["obstacle_in_path"] = clean["obstacle_in_path"]
            shared_state["recommended_side"] = clean["recommended_side"]
            shared_state["speed_scale"] = speed_scale
            shared_state["hold_position"] = hold_position

            frame_path = None
            if logger:
                frame_path = logger.log_frame_b64(
                    image_b64,
                    label="hybrid_gemini",
                    metadata={"odometry": {"x": x, "y": y, "cap": cap}},
                )
                logger.log_event(
                    "gemini_decision",
                    frame=frame_path,
                    decision=clean,
                    cap_cible=nouveau_cap,
                    speed_scale=speed_scale,
                    hold_position=hold_position,
                    odometry={"x": x, "y": y, "cap": cap},
                    distance_from_start_mm=round(distance_from_start_mm(
                        shared_state.get("start_odometry", {"x": x, "y": y, "cap": cap}),
                        {"x": x, "y": y, "cap": cap},
                    ), 1),
                )

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
                    "target_visible": clean["target_visible"],
                    "target_position": clean["target_position"],
                    "obstacle_in_path": clean["obstacle_in_path"],
                    "recommended_side": clean["recommended_side"],
                    "arrival_confirmations": shared_state.get("arrival_confirmations", 0),
                    "odometry": {"x": x, "y": y, "cap": cap}
                })

            print("🧭 Gemini cap={:.1f}° | {} | conf={:.2f}".format(
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
def _infer_hybrid_nn_sync(model, img_t, sens_t):
    with torch.no_grad():
        return model(img_t, sens_t)

async def boucle_nn_reactive(robot, nn_model, sensors_mean, sensors_std,
                             shared_state, broadcast_data, logger=None):
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

    STATE_NAV_NN = "NAV_NN"
    STATE_CLIFF_ESCAPE = "CLIFF_ESCAPE"
    current_state = STATE_NAV_NN
    escape_frame_counter = 0

    while shared_state["nn_active"] and not shared_state["kill_switch"]:
        start_time = loop.time()

        # --- Calcul FPS ---
        t_now = loop.time()
        if last_tick_time > 0:
            dt = t_now - last_tick_time
            current_fps = 1.0 / dt if dt > 0 else 20.0
            fps_ema = 0.9 * fps_ema + 0.1 * current_fps
        last_tick_time = t_now

        # --- Machine à États Finis (FSM) pour l'échappement ---
        if current_state == STATE_NAV_NN:
            if robot.is_cliff_detected or robot.is_picked_up:
                robot.drive_wheels(0.0, 0.0)
                current_state = STATE_CLIFF_ESCAPE
                escape_frame_counter = 0
                continue
        elif current_state == STATE_CLIFF_ESCAPE:
            robot.drive_wheels(-50.0, -20.0)
            escape_frame_counter += 1
            
            if escape_frame_counter >= 40:
                if robot.is_cliff_detected or robot.is_picked_up:
                    pass # Maintien de l'état d'échappement si le danger est toujours présent
                else:
                    robot.drive_wheels(0.0, 0.0)
                    current_state = STATE_NAV_NN
                    escape_frame_counter = 0
            
            await asyncio.sleep(0.05)
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
        sensors_arr = np.array(sensors, dtype=np.float32)

        # Filtrage conditionnel
        if isinstance(nn_model, CozmoNNDiscrete):
            SAFE_SENSOR_INDICES = [2, 3, 6, 7, 8, 9, 10, 11]
            sensors_arr = sensors_arr[SAFE_SENSOR_INDICES]

        # --- 3. Normaliser les capteurs ---
        try:
            sensors_normalized = (sensors_arr - sensors_mean[0]) / sensors_std[0]
        except Exception:
            dim = 8 if isinstance(nn_model, CozmoNNDiscrete) else 12
            sensors_normalized = np.zeros((dim,), dtype=np.float32)

        # --- 4. Lire cap_cible et calculer delta_heading ---
        cap_actuel = robot.pose.rotation.angle_z.degrees
        delta_heading_deg = normalize_angle(shared_state["cap_cible"] - cap_actuel)
        delta_heading_normalized = max(-1.0, min(1.0, delta_heading_deg / 180.0))

        # --- 5. Inférence NN ---
        t_inf_start = time.time()
        nn_left = 0.0
        nn_right = 0.0
        try:
            img_tensor = torch.from_numpy(img_arr).unsqueeze(0).unsqueeze(0)  # (1, 1, 60, 80)

            if isinstance(nn_model, CozmoNNDiscrete):
                sensor_tensor = torch.from_numpy(sensors_normalized).unsqueeze(0)
                outputs = await loop.run_in_executor(None, _infer_hybrid_nn_sync, nn_model, img_tensor, sensor_tensor)
                
                probs = torch.softmax(outputs, dim=1)
                max_prob, predicted_class = torch.max(probs, dim=1)
                action_idx = predicted_class.item()
                prob_val = max_prob.item()

                # Filtre de Confiance Algorithmique
                if action_idx in [1, 2] and prob_val < 0.65:
                    action_idx = 0
                
                speeds = {
                    0: (50.0, 50.0),   # Avant
                    1: (-50.0, 50.0),  # Pivot gauche
                    2: (50.0, -50.0),  # Pivot droite
                    3: (-50.0, -20.0)  # Arrière asymétrique (Anti-ping-pong)
                }
                nn_left, nn_right = speeds.get(action_idx, (0.0, 0.0))
                head_angle = 15.0

                kp = HYBRID_CONFIG["kp_correction"]
                nn_diff = abs(nn_left - nn_right)
                dynamic_kp = kp * max(0.0, 1.0 - (nn_diff / 20.0))
                correction = dynamic_kp * delta_heading_normalized * 50.0
                
                if correction > 0:
                    left_speed = nn_left - correction
                    right_speed = nn_right
                else:
                    left_speed = nn_left
                    right_speed = nn_right + correction

            elif use_v2:
                # Option A : 13 inputs (delta_heading comme 13ème entrée)
                sensor_with_heading = np.append(sensors_normalized, delta_heading_normalized)
                sensor_tensor = torch.from_numpy(sensor_with_heading).unsqueeze(0)  # (1, 13)
                outputs = await loop.run_in_executor(None, _infer_hybrid_nn_sync, nn_model, img_tensor, sensor_tensor)
                left_speed = float(outputs[0, 0].item())
                right_speed = float(outputs[0, 1].item())
                head_angle = float(outputs[0, 2].item())
                nn_left = left_speed
                nn_right = right_speed
            else:
                # Option B : 12 inputs + correction PID sur le cap
                sensor_tensor = torch.from_numpy(sensors_normalized).unsqueeze(0)  # (1, 12)
                outputs = await loop.run_in_executor(None, _infer_hybrid_nn_sync, nn_model, img_tensor, sensor_tensor)
                nn_left = float(outputs[0, 0].item())
                nn_right = float(outputs[0, 1].item())
                head_angle = float(outputs[0, 2].item())

                # Kp dynamique : on laisse la priorité au NN s'il esquive un obstacle
                # Si le NN demande de tourner (différence > 20), on annule le cap Gemini
                kp = HYBRID_CONFIG["kp_correction"]
                nn_diff = abs(nn_left - nn_right)
                
                dynamic_kp = kp * max(0.0, 1.0 - (nn_diff / 20.0))
                
                correction = dynamic_kp * delta_heading_normalized * 50.0
                
                # Sécurité anti-mur : le PID ne peut que ralentir la roue intérieure
                # du virage. Il ne peut pas accélérer une roue pour forcer le cap.
                if correction > 0: # Doit tourner à gauche
                    left_speed = nn_left - correction
                    right_speed = nn_right # On n'accélère pas la droite
                else:              # Doit tourner à droite
                    left_speed = nn_left   # On n'accélère pas la gauche
                    right_speed = nn_right + correction # correction négative, ralentit la droite

        except Exception as e:
            print("Inference error: {}".format(e))
            left_speed = 0.0
            right_speed = 0.0
            head_angle = robot.head_angle.degrees if robot.head_angle else 0.0
            nn_left = 0.0
            nn_right = 0.0

        t_inf_end = time.time()
        inference_ms = (t_inf_end - t_inf_start) * 1000.0
        inf_ms_ema = 0.9 * inf_ms_ema + 0.1 * inference_ms

        # --- 6. Controle d'approche finale / hold strategique ---
        speed_scale = float(shared_state.get("speed_scale", 1.0))
        speed_scale = max(0.0, min(1.0, speed_scale))
        if shared_state.get("hold_position", False):
            left_speed = 0.0
            right_speed = 0.0
        else:
            left_speed *= speed_scale
            right_speed *= speed_scale

        # --- 7. Clip et appliquer les commandes aux roues ---
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
            current_odometry = odometry_snapshot(robot)
            if logger:
                logger.log_event(
                    "nn_tick",
                    nn_raw={"left_speed": round(nn_left, 2), "right_speed": round(nn_right, 2)},
                    final_output={"left_speed": round(left_speed, 2), "right_speed": round(right_speed, 2), "head_angle": round(head_angle, 2)},
                    cap_cible=round(shared_state["cap_cible"], 2),
                    cap_actuel=round(cap_actuel, 2),
                    delta_heading=round(delta_heading_deg, 2),
                    speed_scale=round(speed_scale, 2),
                    hold_position=bool(shared_state.get("hold_position", False)),
                    odometry=current_odometry,
                    distance_from_start_mm=round(distance_from_start_mm(
                        shared_state.get("start_odometry", current_odometry),
                        current_odometry,
                    ), 1),
                )
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
                    "objectif_atteint": shared_state["objectif_atteint"],
                    "target_visible": shared_state.get("target_visible", False),
                    "target_position": shared_state.get("target_position", "unknown"),
                    "obstacle_in_path": shared_state.get("obstacle_in_path", False),
                    "recommended_side": shared_state.get("recommended_side", "none"),
                    "arrival_confirmations": shared_state.get("arrival_confirmations", 0),
                    "speed_scale": round(speed_scale, 2),
                    "hold_position": bool(shared_state.get("hold_position", False))
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
    logger = NavigationRunLogger("hybrid_nn_gemini", commande_utilisateur)

    # --- 1. Trouver et charger le modèle ---
    if model_name is None:
        model_name = find_latest_model()
    if model_name is None:
        logger.summarize("failed_no_model")
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
        logger.summarize("failed_model_load", model_name=model_name)
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
    cap_depart_initial = robot.pose.rotation.angle_z.degrees
    if shared_state is None:
        shared_state = {
            "cap_cible": cap_depart_initial,
            "cap_origine": cap_depart_initial,
            "start_odometry": odometry_snapshot(robot),
            "description_scene": "",
            "strategie": "",
            "objectif_atteint": False,
            "confiance": 0.0,
            "kill_switch": False,
            "nn_active": True,
            "gemini_active": True,
            "arrival_confirmations": 0,
            "speed_scale": 1.0,
            "hold_position": False,
            "target_visible": False,
            "target_position": "unknown",
            "target_size": "unknown",
            "target_touching_bottom": False,
            "obstacle_in_path": False,
            "recommended_side": "none",
        }
    else:
        # S'assurer que les flags de démarrage sont corrects
        shared_state["nn_active"] = True
        shared_state["gemini_active"] = True
        shared_state["kill_switch"] = False
        shared_state["objectif_atteint"] = False
        if "cap_origine" not in shared_state:
            shared_state["cap_origine"] = cap_depart_initial
            shared_state["cap_cible"] = cap_depart_initial
        shared_state["start_odometry"] = odometry_snapshot(robot)
        shared_state["arrival_confirmations"] = 0
        shared_state["speed_scale"] = 1.0
        shared_state["hold_position"] = False
        shared_state["target_visible"] = False
        shared_state["target_position"] = "unknown"
        shared_state["target_size"] = "unknown"
        shared_state["target_touching_bottom"] = False
        shared_state["obstacle_in_path"] = False
        shared_state["recommended_side"] = "none"

    logger.log_event(
        "model_loaded",
        model_name=model_name,
        use_v2=use_v2,
        start_odometry=shared_state.get("start_odometry"),
    )

    # --- 3. Préparer le robot : tête à 0° pour cadrer l'horizon ---
    await robot.set_head_angle(cozmo.util.degrees(0)).wait_for_completed()
    await robot.set_lift_height(1.0).wait_for_completed()

    # --- 4. Lancer les deux boucles en parallèle ---
    tache_gemini = asyncio.create_task(
        boucle_gemini_strategique(robot, commande_utilisateur, shared_state, broadcast_data, logger)
    )
    tache_nn = asyncio.create_task(
        boucle_nn_reactive(robot, nn_model, sensors_mean, sensors_std,
                           shared_state, broadcast_data, logger)
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
    stop_reason = "ended"

    while True:
        await asyncio.sleep(0.5)

        # Vérifier objectif atteint
        if shared_state["objectif_atteint"]:
            stop_reason = "success"
            if broadcast_data:
                await broadcast_data({
                    "type": "terminal_log",
                    "message": "🎯 Objectif atteint ! Arrêt de la navigation hybride.",
                    "source": "system"
                })
            break

        # Vérifier kill_switch
        if shared_state["kill_switch"]:
            stop_reason = "stopped"
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
            stop_reason = "timeout"
            if broadcast_data:
                await broadcast_data({
                    "type": "terminal_log",
                    "message": "⏱ Timeout ({:.0f}s). Arrêt de la navigation hybride.".format(timeout),
                    "source": "system"
                })
            break

        # Vérifier si les deux boucles sont mortes
        if tache_gemini.done() and tache_nn.done():
            stop_reason = "ended"
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

    final_odometry = odometry_snapshot(robot)
    status = stop_reason
    logger.summarize(
        status,
        model_name=model_name,
        final_odometry=final_odometry,
        distance_from_start_mm=round(distance_from_start_mm(
            shared_state.get("start_odometry", final_odometry),
            final_odometry,
        ), 1),
        arrival_confirmations=shared_state.get("arrival_confirmations", 0),
    )

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
