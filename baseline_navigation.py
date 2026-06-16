"""
baseline_navigation.py
======================
Baseline fiable pour le test "tour Eiffel + boite" sans NN.

Le principe est volontairement conservateur:
  Stop -> Look -> Gemini scene contract -> petite action -> Stop -> Look.
Gemini percoit la cible et l'obstacle; le controleur local execute seulement
des mouvements courts et journalises.
"""

import asyncio
import base64
import io
import json
import os
import time

import cozmo
import numpy as np
from dotenv import load_dotenv

from nav_benchmark import NavigationRunLogger, distance_from_start_mm, odometry_snapshot

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")


BASELINE_CONFIG = {
    "timeout_s": 300.0,
    "max_steps": 80,
    "frame_settle_s": 0.35,
    "move_speed_mmps": 45,
    "turn_deg_large": 42,
    "turn_deg_small": 13,
    "advance_far_mm": 140,
    "advance_mid_mm": 90,
    "advance_near_mm": 45,
    "arrival_confidence": 0.74,
    "arrival_confirmations": 2,
}


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


def _normalize_side(value):
    value = str(value or "none").strip().lower()
    aliases = {"gauche": "left", "droite": "right", "aucun": "none"}
    value = aliases.get(value, value)
    if value not in ("left", "right", "none"):
        return "none"
    return value


def sanitize_scene_decision(raw):
    raw = raw or {}
    return {
        "target_visible": _coerce_bool(raw.get("target_visible")),
        "target_position": _normalize_position(raw.get("target_position")),
        "target_size": _normalize_size(raw.get("target_size")),
        "target_touching_bottom": _coerce_bool(raw.get("target_touching_bottom")),
        "obstacle_in_path": _coerce_bool(raw.get("obstacle_in_path")),
        "obstacle_position": _normalize_position(raw.get("obstacle_position")),
        "recommended_side": _normalize_side(raw.get("recommended_side")),
        "arrival_confidence": max(0.0, min(1.0, _coerce_float(raw.get("arrival_confidence"), 0.0))),
        "safe_to_advance": _coerce_bool(raw.get("safe_to_advance"), True),
        "diagnostic": str(raw.get("diagnostic", "")),
        "failure_risk": str(raw.get("failure_risk", "none")),
        "raw": raw,
    }


def capture_snapshot_b64(robot):
    latest = robot.world.latest_image
    if latest is None or latest.raw_image is None:
        return None
    pil_img = latest.raw_image.resize((320, 240))
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=75)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


async def appel_gemini_scene_async(image_b64, objective, odometry, previous_action, history):
    import aiohttp

    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        "gemini-3.1-flash-lite:generateContent?key={}".format(API_KEY)
    )

    system_prompt = (
        "Tu es la perception visuelle d'un robot Cozmo dans un test reproductible. "
        "Objectif typique: rejoindre une tour Eiffel LEGO de 30 cm, avec parfois "
        "une boite a chaussures entre le robot et la cible. Tu dois detecter la cible, "
        "dire si un obstacle bloque le chemin direct, et estimer si le robot est arrive. "
        "Ne donne pas de longues explications. Reponds uniquement avec ce JSON strict:\n"
        "{\n"
        '  "target_visible": true,\n'
        '  "target_position": "left|center|right|unknown",\n'
        '  "target_size": "small|medium|large|very_large|unknown",\n'
        '  "target_touching_bottom": false,\n'
        '  "obstacle_in_path": false,\n'
        '  "obstacle_position": "left|center|right|none|unknown",\n'
        '  "recommended_side": "left|right|none",\n'
        '  "arrival_confidence": 0.0,\n'
        '  "safe_to_advance": true,\n'
        '  "failure_risk": "none|collision|lost_target|too_close|wall",\n'
        '  "diagnostic": "phrase courte"\n'
        "}\n"
        "Regles: obstacle_in_path=true si la boite occupe le couloir direct devant Cozmo. "
        "recommended_side indique le cote qui semble le plus libre pour contourner. "
        "arrival_confidence ne doit depasser 0.74 que si la cible est tres proche; "
        "target_touching_bottom=true seulement si la base de la cible touche le bord bas de l'image."
    )

    user_text = (
        "Objectif: {}. Odometrie: x={:.1f} mm, y={:.1f} mm, cap={:.1f} deg. "
        "Derniere action: {}. Analyse l'image."
    ).format(
        objective,
        odometry.get("x", 0.0),
        odometry.get("y", 0.0),
        odometry.get("cap", 0.0),
        previous_action,
    )

    user_parts = [{"text": user_text}]
    if image_b64:
        user_parts.append({"inlineData": {"mimeType": "image/jpeg", "data": image_b64}})

    payload = {
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": list(history) + [{"role": "user", "parts": user_parts}],
    }

    async with aiohttp.ClientSession() as session:
        for attempt in range(3):
            try:
                async with session.post(url, headers={"Content-Type": "application/json"}, json=payload, timeout=15) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                    raw = data["candidates"][0]["content"]["parts"][0]["text"]
                    raw = raw.replace("```json", "").replace("```", "").strip()
                    return json.loads(raw)
            except aiohttp.ClientResponseError as exc:
                if exc.status == 429 and attempt < 2:
                    await asyncio.sleep(2 * (attempt + 1))
                else:
                    print("Gemini baseline error: {}".format(exc))
                    return None
            except Exception as exc:
                print("Gemini baseline error: {}".format(exc))
                return None
    return None


async def _watch_cliff(robot):
    while True:
        if robot.is_cliff_detected:
            return True
        await asyncio.sleep(0.05)


async def _execute_motion(robot, motion, logger, action_payload):
    logger.log_event("action_started", action=action_payload, odometry=odometry_snapshot(robot))
    move_task = asyncio.ensure_future(motion.wait_for_completed())
    cliff_task = asyncio.ensure_future(_watch_cliff(robot))
    done, pending = await asyncio.wait({move_task, cliff_task}, return_when=asyncio.FIRST_COMPLETED)

    for task in pending:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    if cliff_task in done:
        robot.abort_all_actions()
        await robot.drive_straight(
            cozmo.util.distance_mm(-35),
            cozmo.util.speed_mmps(50),
        ).wait_for_completed()
        logger.log_event("cliff_interrupt", action=action_payload, odometry=odometry_snapshot(robot))
        return False

    logger.log_event("action_completed", action=action_payload, odometry=odometry_snapshot(robot))
    return True


def _turn_motion(robot, side, degrees):
    signed = -abs(degrees) if side == "right" else abs(degrees)
    return robot.turn_in_place(cozmo.util.degrees(signed))


def _plan_action(decision, last_target_side, avoidance_side, avoidance_steps_left):
    cfg = BASELINE_CONFIG

    if avoidance_steps_left > 0:
        side = avoidance_side or "right"
        opposite = "left" if side == "right" else "right"
        sequence = {
            4: ("turn", side, cfg["turn_deg_large"]),
            3: ("advance", None, cfg["advance_far_mm"]),
            2: ("advance", None, cfg["advance_far_mm"]),
            1: ("turn", opposite, cfg["turn_deg_large"] - 8),
        }
        return sequence.get(avoidance_steps_left, ("advance", None, cfg["advance_mid_mm"]))

    if decision["obstacle_in_path"]:
        side = decision["recommended_side"] if decision["recommended_side"] != "none" else "right"
        return ("start_avoidance", side, 4)

    if decision["target_visible"]:
        position = decision["target_position"]
        if position == "left":
            return ("turn", "left", cfg["turn_deg_small"])
        if position == "right":
            return ("turn", "right", cfg["turn_deg_small"])

        if not decision["safe_to_advance"]:
            return ("turn", last_target_side or "right", cfg["turn_deg_small"])

        if decision["target_size"] in ("large", "very_large"):
            return ("advance", None, cfg["advance_near_mm"])
        if decision["target_size"] == "medium":
            return ("advance", None, cfg["advance_mid_mm"])
        return ("advance", None, cfg["advance_far_mm"])

    return ("turn", last_target_side or "right", cfg["turn_deg_small"] + 7)


async def lancer_benchmark_eiffel_baseline(robot, objective, broadcast_data=None, shared_state=None):
    """
    Baseline benchmark entry point. It does not load or use any NN.
    """
    logger = NavigationRunLogger("baseline_stop_look_plan_act", objective)
    cfg = BASELINE_CONFIG
    start_time = time.time()
    start_odometry = odometry_snapshot(robot)
    history = []
    previous_action = "start"
    last_target_side = None
    avoidance_side = None
    avoidance_steps_left = 0
    arrival_hits = 0
    status = "timeout"

    if shared_state is None:
        shared_state = {"kill_switch": False}

    if broadcast_data:
        await broadcast_data({
            "type": "terminal_log",
            "message": "Baseline benchmark demarre. Logs: {}".format(logger.run_dir),
            "source": "system",
        })

    try:
        await robot.set_head_angle(cozmo.util.degrees(0)).wait_for_completed()
        await robot.set_lift_height(1.0).wait_for_completed()
    except Exception:
        pass

    for step in range(1, cfg["max_steps"] + 1):
        if shared_state.get("kill_switch"):
            status = "stopped"
            break

        if time.time() - start_time > cfg["timeout_s"]:
            status = "timeout"
            break

        try:
            await robot.stop_all_motors()
        except Exception:
            pass
        await asyncio.sleep(cfg["frame_settle_s"])

        odom = odometry_snapshot(robot)
        image_b64 = capture_snapshot_b64(robot)
        frame_path = logger.log_frame_b64(
            image_b64,
            label="look_{:03d}".format(step),
            metadata={"odometry": odom, "step": step},
        )

        raw_decision = await appel_gemini_scene_async(
            image_b64,
            objective,
            odom,
            previous_action,
            history,
        )

        if raw_decision is None:
            logger.log_event("gemini_failure", step=step, odometry=odom)
            await asyncio.sleep(2.0)
            continue

        decision = sanitize_scene_decision(raw_decision)
        distance_mm = distance_from_start_mm(start_odometry, odom)
        logger.log_event(
            "scene_decision",
            step=step,
            frame=frame_path,
            decision=decision,
            odometry=odom,
            distance_from_start_mm=round(distance_mm, 1),
        )

        history.append({
            "role": "user",
            "parts": [{"text": "Step {} odometry {}".format(step, json.dumps(odom))}],
        })
        history.append({
            "role": "model",
            "parts": [{"text": json.dumps(raw_decision, ensure_ascii=False)}],
        })
        while len(history) > 6:
            history.pop(0)

        if decision["target_position"] in ("left", "right"):
            last_target_side = decision["target_position"]

        if (
            decision["target_visible"]
            and decision["target_touching_bottom"]
            and decision["arrival_confidence"] >= cfg["arrival_confidence"]
        ):
            arrival_hits += 1
        else:
            arrival_hits = 0

        if arrival_hits >= cfg["arrival_confirmations"]:
            status = "success"
            logger.log_event("arrival_confirmed", step=step, decision=decision, odometry=odom)
            break

        action = _plan_action(decision, last_target_side, avoidance_side, avoidance_steps_left)
        kind, side, value = action

        if kind == "start_avoidance":
            avoidance_side = side
            avoidance_steps_left = int(value)
            logger.log_event("avoidance_started", step=step, side=avoidance_side)
            continue

        if avoidance_steps_left > 0:
            avoidance_steps_left -= 1

        if kind == "turn":
            motion = _turn_motion(robot, side, value)
            action_payload = {"kind": "turn", "side": side, "degrees": value}
        else:
            motion = robot.drive_straight(
                cozmo.util.distance_mm(value),
                cozmo.util.speed_mmps(cfg["move_speed_mmps"]),
            )
            action_payload = {"kind": "advance", "distance_mm": value}

        previous_action = json.dumps(action_payload, sort_keys=True)
        ok = await _execute_motion(robot, motion, logger, action_payload)
        if not ok:
            previous_action = "cliff interrupt and emergency backup"

        if broadcast_data:
            await broadcast_data({
                "type": "ai_cot",
                "etape": "Baseline step {}".format(step),
                "diagnostic": decision["diagnostic"],
                "action": action_payload["kind"],
                "valeur": action_payload.get("distance_mm", action_payload.get("degrees", "")),
                "raw": decision,
                "odometry": odom,
                "frame": image_b64,
            })

    try:
        await robot.stop_all_motors()
        await robot.set_head_angle(cozmo.util.degrees(30)).wait_for_completed()
    except Exception:
        pass

    final_odometry = odometry_snapshot(robot)
    summary = logger.summarize(
        status,
        final_odometry=final_odometry,
        distance_from_start_mm=round(distance_from_start_mm(start_odometry, final_odometry), 1),
        arrival_confirmations=arrival_hits,
    )

    if broadcast_data:
        await broadcast_data({
            "type": "terminal_log",
            "message": "Baseline benchmark termine: {} | logs {}".format(status, logger.run_dir),
            "source": "system",
        })

    return summary
