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
import time
import numpy as np
import speech_recognition as sr
from PIL import Image, ImageDraw, ImageFont
from dotenv import load_dotenv

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# Module de navigation Waypoint (Stop, Look, Think, Act + Fail-Safe)
from navigation import lancer_exploration

# Module de navigation hybride Phase 4 (NN + Gemini)
from hybrid_navigation import lancer_exploration_hybride, find_latest_model

# Import conditionnel de PyTorch pour le mode autonome NN
TORCH_AVAILABLE = False
try:
    import torch
    from model import CozmoNN
    TORCH_AVAILABLE = True
except ImportError:
    pass

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

# Endpoint pour récupérer la liste des modèles disponibles
@app.get("/api/models")
def get_models():
    models_dir = "models"
    if not os.path.exists(models_dir):
        return []
    
    models_list = []
    for f in os.listdir(models_dir):
        if f.endswith(".pt"):
            filepath = os.path.join(models_dir, f)
            stat_info = os.stat(filepath)
            
            # Read metadata if exists
            metadata_path = os.path.join(models_dir, f.replace(".pt", ".json"))
            display_name = f[:-3]
            version = ""
            parameters = ""
            
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r') as mf:
                        meta = json.load(mf)
                        display_name = meta.get("displayName", display_name)
                        version = meta.get("version", "")
                        parameters = meta.get("parameters", "")
                except Exception:
                    pass
                    
            models_list.append({
                "name": f[:-3],  # Nom sans .pt
                "displayName": display_name,
                "version": version,
                "parameters": parameters,
                "filename": f,
                "size_bytes": stat_info.st_size,
                "modified_time": stat_info.st_mtime
            })
            
    # Tri par date de modification décroissante (le plus récent en premier)
    models_list.sort(key=lambda x: x["modified_time"], reverse=True)
    return models_list

active_websockets = []
command_queue = asyncio.Queue()

# --- VARIABLES GLOBALES DU MODE ENTRAÎNEMENT ---
global_robot = None
training_mode_active = False
is_recording = False
training_frames = []
training_sensors = []
training_actions = []
training_timestamps = []
current_left_speed = 0.0
current_right_speed = 0.0
current_head_angle = 30.0
last_drive_command_time = 0.0
current_session_name = ""
recording_task = None

# --- VARIABLES GLOBALES DU MODE NN ---
nn_active = False
nn_task = None
nn_model_name = ""
nn_model = None
sensors_mean = None
sensors_std = None

# --- VARIABLES GLOBALES DU MODE HYBRIDE ---
hybrid_active = False
hybrid_task = None
hybrid_shared_state = None

def load_model_and_stats_sync(model_name):
    global nn_model, sensors_mean, sensors_std
    try:
        # Strip .pt if present
        model_key = model_name
        if model_key.endswith(".pt"):
            model_key = model_key[:-3]
            
        if model_key.startswith("cozmo_nn_"):
            suffix = model_key[len("cozmo_nn_"):]
        else:
            suffix = model_key
            
        model_path = os.path.join("models", f"{model_key}.pt")
        norm_stats_path = os.path.join("models", f"norm_stats_{suffix}.json")
        
        if not os.path.exists(model_path):
            print(f"Model file not found: {model_path}")
            return False
            
        if not os.path.exists(norm_stats_path):
            print(f"Norm stats file not found: {norm_stats_path}")
            return False
            
        # Load stats
        with open(norm_stats_path, 'r') as f:
            stats = json.load(f)
        sensors_mean = np.array(stats['mean'], dtype=np.float32)
        sensors_std = np.array(stats['std'], dtype=np.float32)
        
        # Instantiate model
        nn_model = CozmoNN()
        state_dict = torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)
        nn_model.load_state_dict(state_dict)
        nn_model.eval()
        
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

async def start_nn_mode(robot, model_name):
    global nn_active, nn_task, nn_model_name
    if not TORCH_AVAILABLE:
        await broadcast_data({"type": "terminal_log", "message": "❌ Erreur : PyTorch n'est pas disponible.", "source": "system"})
        return
        
    if nn_active:
        return
        
    # Stop other conflicting modes
    global training_mode_active
    if training_mode_active:
        training_mode_active = False
        await stop_recording_and_save(robot)
        
    nn_model_name = model_name
    nn_active = True
    
    loop = asyncio.get_event_loop()
    try:
        await broadcast_data({"type": "terminal_log", "message": f"🧠 Chargement du modèle {model_name}...", "source": "system"})
        success = await loop.run_in_executor(None, load_model_and_stats_sync, model_name)
        if not success:
            nn_active = False
            await broadcast_data({"type": "terminal_log", "message": f"❌ Échec du chargement du modèle {model_name}", "source": "system"})
            return
            
        nn_task = asyncio.create_task(nn_inference_loop(robot))
        await broadcast_data({"type": "terminal_log", "message": f"▶️ Mode NN démarré avec {model_name}", "source": "system"})
        # Also broadcast initial status
        await broadcast_data({
            "type": "nn_status",
            "active": True,
            "model_name": nn_model_name,
            "fps": 20.0,
            "inference_ms": 0.0,
            "left_speed": 0.0,
            "right_speed": 0.0,
            "head_angle": robot.head_angle.degrees if robot.head_angle else 0.0,
            "cliff_events": 0
        })
    except Exception as e:
        nn_active = False
        await broadcast_data({"type": "terminal_log", "message": f"❌ Erreur de chargement : {str(e)}", "source": "system"})

async def stop_nn_mode(robot):
    global nn_active, nn_task
    if not nn_active:
        return
    nn_active = False
    if robot:
        try:
            await robot.stop_all_motors()
        except Exception:
            pass
            
    if nn_task:
        try:
            await nn_task
        except Exception:
            pass
        nn_task = None
        
    await broadcast_data({"type": "terminal_log", "message": "⏹ Mode NN arrêté.", "source": "system"})
    await broadcast_data({
        "type": "nn_status",
        "active": False,
        "model_name": "",
        "fps": 0.0,
        "inference_ms": 0.0,
        "left_speed": 0.0,
        "right_speed": 0.0,
        "head_angle": 0.0,
        "cliff_events": 0
    })

# =====================================================================
# --- FONCTIONS DU MODE HYBRIDE (Phase 4) ---
# =====================================================================
async def start_hybrid_mode(robot, commande_utilisateur, model_name=None):
    """Démarre le mode exploration hybride NN + Gemini."""
    global hybrid_active, hybrid_task, hybrid_shared_state
    global training_mode_active, nn_active

    if hybrid_active:
        await broadcast_data({"type": "terminal_log", "message": "⚠️ Mode hybride déjà actif.", "source": "system"})
        return

    # Arrêter les modes conflictuels
    if training_mode_active:
        training_mode_active = False
        await stop_recording_and_save(robot)
    if nn_active:
        await stop_nn_mode(robot)

    hybrid_active = True
    # Créer le shared_state immédiatement pour que le kill switch fonctionne
    hybrid_shared_state = {
        "cap_cible": 0.0,
        "description_scene": "",
        "strategie": "",
        "objectif_atteint": False,
        "confiance": 0.0,
        "kill_switch": False,
        "nn_active": True,
        "gemini_active": True,
    }
    await broadcast_data({"type": "terminal_log", "message": "🚀 Démarrage du mode hybride...", "source": "system"})

    async def _run_hybrid():
        global hybrid_active, hybrid_shared_state
        try:
            await lancer_exploration_hybride(
                robot, commande_utilisateur, broadcast_data,
                model_name=model_name, shared_state=hybrid_shared_state
            )
        except Exception as e:
            await broadcast_data({"type": "terminal_log", "message": "❌ Erreur mode hybride : " + str(e), "source": "system"})
        finally:
            hybrid_active = False
            hybrid_shared_state = None
            await broadcast_data({"type": "hybrid_stopped"})
            await broadcast_data({"type": "terminal_log", "message": "⏹ Mode hybride arrêté.", "source": "system"})

    hybrid_task = asyncio.create_task(_run_hybrid())


async def stop_hybrid_mode():
    """Arrête le mode hybride en activant le kill switch."""
    global hybrid_active, hybrid_task, hybrid_shared_state

    if not hybrid_active:
        return

    # Activer le kill switch pour arrêter les deux boucles
    if hybrid_shared_state is not None:
        hybrid_shared_state["kill_switch"] = True

    # Attendre la fin propre
    if hybrid_task is not None:
        try:
            await asyncio.wait_for(hybrid_task, timeout=5.0)
        except asyncio.TimeoutError:
            hybrid_task.cancel()
            try:
                await hybrid_task
            except (asyncio.CancelledError, Exception):
                pass
        except Exception:
            pass
        hybrid_task = None

    hybrid_active = False
    hybrid_shared_state = None

    if global_robot:
        try:
            await global_robot.stop_all_motors()
        except Exception:
            pass

    await broadcast_data({"type": "hybrid_stopped"})


async def nn_inference_loop(robot):
    global nn_active, nn_model, sensors_mean, sensors_std, nn_model_name
    loop = asyncio.get_event_loop()
    
    fps_ema = 20.0
    inf_ms_ema = 0.0
    last_tick_time = 0.0
    tick_count = 0
    cliff_events = 0
    watchdog_counter = 0

    while nn_active:
        start_time = loop.time()
        
        # Calculate FPS
        t_now = loop.time()
        if last_tick_time > 0:
            dt = t_now - last_tick_time
            current_fps = 1.0 / dt if dt > 0 else 20.0
            fps_ema = 0.9 * fps_ema + 0.1 * current_fps
        last_tick_time = t_now
        
        # Sécurité anti-chute (cliff override)
        if robot.is_cliff_detected:
            cliff_events += 1
            await robot.stop_all_motors()
            await robot.drive_straight(cozmo.util.distance_mm(-50), cozmo.util.speed_mmps(50)).wait_for_completed()
            await asyncio.sleep(0.5)
            continue
            
        # 1. Capture image (80x60 grayscale, normalisée [0,1])
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
            
        # 2. Capture sensory vector
        sensors = get_sensor_vector(robot)
        
        # 3. Normaliser les capteurs
        try:
            sensors_normalized = (np.array(sensors, dtype=np.float32) - sensors_mean[0]) / sensors_std[0]
        except Exception as e:
            print(f"Normalization error: {e}")
            sensors_normalized = np.zeros((12,), dtype=np.float32)
            
        # 4. Inférence (forward pass)
        t_inf_start = time.time()
        try:
            img_tensor = torch.from_numpy(img_arr).unsqueeze(0).unsqueeze(0)      # (1, 1, 60, 80)
            sensor_tensor = torch.from_numpy(sensors_normalized).unsqueeze(0)    # (1, 12)
            
            with torch.no_grad():
                outputs = nn_model(img_tensor, sensor_tensor)
                left_speed = float(outputs[0, 0].item())
                right_speed = float(outputs[0, 1].item())
                head_angle = float(outputs[0, 2].item())
        except Exception as e:
            print(f"Inference error: {e}")
            left_speed = 0.0
            right_speed = 0.0
            head_angle = robot.head_angle.degrees if robot.head_angle else 0.0
            
        t_inf_end = time.time()
        inference_ms = (t_inf_end - t_inf_start) * 1000.0
        inf_ms_ema = 0.9 * inf_ms_ema + 0.1 * inference_ms
        
        # Watchdog
        if inference_ms > 100.0:
            watchdog_counter += 1
            print(f"⚠️ Warning: Inference took {inference_ms:.1f}ms (> 100ms)")
            await broadcast_data({"type": "terminal_log", "message": f"⚠️ Inférence lente : {inference_ms:.1f}ms ({watchdog_counter}/5)", "source": "system"})
            if watchdog_counter >= 5:
                await broadcast_data({"type": "terminal_log", "message": "🚨 Arrêt d'urgence : 5 inférences lentes consécutives !", "source": "system"})
                nn_active = False
                await robot.stop_all_motors()
                await broadcast_data({
                    "type": "nn_status",
                    "active": False,
                    "model_name": "",
                    "fps": 0.0,
                    "inference_ms": 0.0,
                    "left_speed": 0.0,
                    "right_speed": 0.0,
                    "head_angle": 0.0,
                    "cliff_events": cliff_events
                })
                break
        else:
            watchdog_counter = 0
            
        # 5. Appliquer les commandes aux roues
        try:
            left_speed = max(-150.0, min(150.0, left_speed))
            right_speed = max(-150.0, min(150.0, right_speed))
            await robot.drive_wheels(left_speed, right_speed)
        except Exception as e:
            print(f"Error driving wheels: {e}")
            
        # 6. Ajuster l'angle de la tête
        try:
            current_head = robot.head_angle.degrees if robot.head_angle else 0.0
            if abs(head_angle - current_head) > 2.0:
                robot.set_head_angle(cozmo.util.degrees(head_angle), in_parallel=True)
        except Exception as e:
            print(f"Error setting head angle: {e}")
            
        # 7. Broadcast vers le dashboard les décisions du NN (~4Hz)
        tick_count += 1
        if tick_count % 5 == 0:
            await broadcast_data({
                "type": "nn_status",
                "active": True,
                "model_name": nn_model_name,
                "fps": round(fps_ema, 1),
                "inference_ms": round(inf_ms_ema, 1),
                "left_speed": round(left_speed, 1),
                "right_speed": round(right_speed, 1),
                "head_angle": round(head_angle, 1),
                "cliff_events": cliff_events
            })
            
        # Sleep pour maintenir 20Hz
        elapsed = loop.time() - start_time
        sleep_time = max(0.001, 0.05 - elapsed)
        await asyncio.sleep(sleep_time)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global training_mode_active, current_left_speed, current_right_speed, last_drive_command_time, current_head_angle, nn_active
    await websocket.accept()
    active_websockets.append(websocket)
    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type")
            if msg_type == "command":
                await command_queue.put(data.get("command"))
            elif msg_type == "training_mode":
                training_mode_active = data.get("active", False)
                if training_mode_active and nn_active:
                    await stop_nn_mode(global_robot)
                if not training_mode_active and is_recording:
                    await stop_recording_and_save(global_robot)
            elif msg_type == "training_start":
                await start_recording(global_robot)
            elif msg_type == "training_stop":
                await stop_recording_and_save(global_robot)
            elif msg_type == "training_drive":
                current_left_speed = float(data.get("left", 0.0))
                current_right_speed = float(data.get("right", 0.0))
                last_drive_command_time = asyncio.get_event_loop().time()
                if global_robot:
                    try:
                        # Sécurité anti-chute en mode entraînement
                        if global_robot.is_cliff_detected:
                            is_backward = (current_left_speed <= 0 and current_right_speed <= 0)
                            is_turn_in_place = (current_left_speed * current_right_speed < 0)
                            if not (is_backward or is_turn_in_place):
                                current_left_speed = 0.0
                                current_right_speed = 0.0
                                await global_robot.stop_all_motors()
                        await global_robot.drive_wheels(current_left_speed, current_right_speed)
                    except Exception:
                        pass
            elif msg_type == "training_head":
                angle = float(data.get("angle", 0.0))
                current_head_angle = angle
                if global_robot:
                    try:
                        global_robot.set_head_angle(cozmo.util.degrees(angle), in_parallel=True)
                    except Exception:
                        pass
            elif msg_type == "nn_start":
                model_name = data.get("model")
                if global_robot:
                    await start_nn_mode(global_robot, model_name)
            elif msg_type == "nn_stop":
                if global_robot:
                    await stop_nn_mode(global_robot)
            elif msg_type == "hybrid_start":
                objective = data.get("objective", "")
                model_name = data.get("model", "") or None
                if global_robot and objective:
                    await start_hybrid_mode(global_robot, objective, model_name)
            elif msg_type == "hybrid_stop":
                await stop_hybrid_mode()
    except WebSocketDisconnect:
        active_websockets.remove(websocket)
        if not active_websockets:
            if nn_active:
                await stop_nn_mode(global_robot)
            if hybrid_active:
                await stop_hybrid_mode()

async def broadcast_data(data: dict):
    for ws in list(active_websockets):
        try:
            await ws.send_json(data)
        except Exception:
            pass

# =====================================================================
# --- FONCTIONS DU MODE ENTRAÎNEMENT ---
# =====================================================================
def get_sensor_vector(robot):
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

async def training_recording_loop(robot):
    global is_recording, training_frames, training_sensors, training_actions, training_timestamps
    loop = asyncio.get_event_loop()
    
    while is_recording:
        start_time = loop.time()
        
        # 1. Capture image
        try:
            latest = robot.world.latest_image
            if latest is not None and latest.raw_image is not None:
                def process_image(img):
                    gray_img = img.convert('L').resize((80, 60))
                    return np.array(gray_img, dtype=np.uint8)
                img_arr = await loop.run_in_executor(None, process_image, latest.raw_image)
            else:
                img_arr = np.zeros((60, 80), dtype=np.uint8)
        except Exception:
            img_arr = np.zeros((60, 80), dtype=np.uint8)
            
        # 2. Capture sensory vector
        sensors = get_sensor_vector(robot)
        
        # 3. Capture actions
        actions = [
            float(current_left_speed),
            float(current_right_speed),
            float(current_head_angle)
        ]
        
        # 4. Save to list
        training_frames.append(img_arr)
        training_sensors.append(sensors)
        training_actions.append(actions)
        training_timestamps.append(time.time())
        
        # Sleep to maintain 20Hz (50ms interval)
        elapsed = loop.time() - start_time
        sleep_time = max(0.001, 0.05 - elapsed)
        await asyncio.sleep(sleep_time)

async def start_recording(robot):
    global is_recording, training_frames, training_sensors, training_actions, training_timestamps
    global current_session_name, recording_task
    if is_recording or robot is None:
        return
    
    training_frames = []
    training_sensors = []
    training_actions = []
    training_timestamps = []
    
    import datetime
    now_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    current_session_name = f"session_{now_str}"
    is_recording = True
    
    recording_task = asyncio.create_task(training_recording_loop(robot))
    print(f"⏺ Début de l'enregistrement de la session {current_session_name}")

async def stop_recording_and_save(robot):
    global is_recording, recording_task
    if not is_recording:
        return
    
    is_recording = False
    if recording_task:
        try:
            await recording_task
        except Exception:
            pass
        recording_task = None
    
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, stop_and_save_sync)

def stop_and_save_sync():
    global training_frames, training_sensors, training_actions, training_timestamps, current_session_name
    if not training_timestamps:
        print("⚠️ Aucun enregistrement à sauvegarder.")
        return
    
    os.makedirs("training_data", exist_ok=True)
    filepath = os.path.join("training_data", f"{current_session_name}.npz")
    
    frames_arr = np.array(training_frames, dtype=np.uint8)
    sensors_arr = np.array(training_sensors, dtype=np.float32)
    actions_arr = np.array(training_actions, dtype=np.float32)
    timestamps_arr = np.array(training_timestamps, dtype=np.float64)
    
    np.savez_compressed(
        filepath,
        frames=frames_arr,
        sensors=sensors_arr,
        actions=actions_arr,
        timestamps=timestamps_arr
    )
    
    size_mb = os.path.getsize(filepath) / (1024.0 * 1024.0)
    print(f"⏹ Session {current_session_name} sauvegardée avec succès: {filepath} ({size_mb:.2f} MB, {len(timestamps_arr)} frames)")
    
    training_frames = []
    training_sensors = []
    training_actions = []
    training_timestamps = []

async def safety_timeout_loop():
    global current_left_speed, current_right_speed
    while True:
        await asyncio.sleep(0.05)
        if training_mode_active:
            now = asyncio.get_event_loop().time()
            if now - last_drive_command_time > 0.2:
                if current_left_speed != 0.0 or current_right_speed != 0.0:
                    current_left_speed = 0.0
                    current_right_speed = 0.0
                    try:
                        if global_robot:
                            global_robot.drive_wheels(0.0, 0.0)
                    except Exception:
                        pass

async def stream_training_status():
    while True:
        await asyncio.sleep(0.5)
        if training_mode_active:
            if is_recording and len(training_timestamps) > 0:
                duration = training_timestamps[-1] - training_timestamps[0]
            else:
                duration = 0.0
            
            n_samples = len(training_timestamps)
            raw_bytes = n_samples * (60 * 80 * 1 + 12 * 4 + 3 * 4 + 8)
            size_mb = round(raw_bytes / (1024.0 * 1024.0), 2)
            
            await broadcast_data({
                "type": "training_status",
                "recording": is_recording,
                "session_name": current_session_name,
                "frame_count": n_samples,
                "duration_s": round(duration, 2),
                "file_size_mb": size_mb
            })

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
                left_speed = 0.0
                right_speed = 0.0
                try:
                    if robot.left_wheel_speed is not None:
                        left_speed = robot.left_wheel_speed.speed_mmps
                except Exception:
                    pass
                try:
                    if robot.right_wheel_speed is not None:
                        right_speed = robot.right_wheel_speed.speed_mmps
                except Exception:
                    pass
                
                await broadcast_data({
                    "type": "state_update",
                    "odometry": {
                        "x": robot.pose.position.x,
                        "y": robot.pose.position.y,
                        "cap": robot.pose.rotation.angle_z.degrees
                    },
                    "wheels": {
                        "left": left_speed,
                        "right": right_speed
                    },
                    "cliff_detected": robot.is_cliff_detected
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
    global global_robot
    global_robot = robot
    robot.camera.image_stream_enabled = True
    await asyncio.sleep(1.0)
    await robot.set_head_angle(cozmo.util.degrees(30)).wait_for_completed()
    try:
        await robot.set_lift_height(1.0).wait_for_completed()
    except Exception as e:
        print(f"Error setting initial lift height: {e}")
    print("⚙️ Connexion établie.")
    loop = asyncio.get_event_loop()

    # Démarrage des tâches de fond
    asyncio.create_task(start_server())
    asyncio.create_task(stream_video(robot))
    asyncio.create_task(stream_telemetry(robot))
    asyncio.create_task(terminal_input_task(loop))
    asyncio.create_task(safety_timeout_loop())
    asyncio.create_task(stream_training_status())
    
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

            # Phase 4 : routage automatique vers le mode hybride si un modèle NN est disponible
            latest_model = find_latest_model()
            if latest_model and TORCH_AVAILABLE:
                await robot.say_text("Mode exploration hybride activé.", in_parallel=True).wait_for_completed()
                await broadcast_data({"type": "terminal_log", "message": "🚀 Mode hybride NN + Gemini activé.", "source": "system"})
                await start_hybrid_mode(robot, commande_complete)
            else:
                await robot.say_text("Mode exploration activé.", in_parallel=True).wait_for_completed()
                await broadcast_data({"type": "terminal_log", "message": "Mode exploration classique activé (aucun modèle NN disponible).", "source": "system"})
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