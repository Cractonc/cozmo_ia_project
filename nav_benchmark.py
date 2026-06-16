import base64
import json
import os
import re
import time
from datetime import datetime


def _safe_slug(value):
    value = str(value or "").strip().lower()
    value = re.sub(r"[^a-z0-9_-]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "run"


def odometry_snapshot(robot):
    """Return a serializable odometry snapshot without assuming all SDK fields exist."""
    try:
        pose = robot.pose
        return {
            "x": float(pose.position.x),
            "y": float(pose.position.y),
            "cap": float(pose.rotation.angle_z.degrees),
        }
    except Exception:
        return {"x": 0.0, "y": 0.0, "cap": 0.0}


def distance_from_start_mm(start_odometry, current_odometry):
    dx = float(current_odometry.get("x", 0.0)) - float(start_odometry.get("x", 0.0))
    dy = float(current_odometry.get("y", 0.0)) - float(start_odometry.get("y", 0.0))
    return (dx * dx + dy * dy) ** 0.5


class NavigationRunLogger:
    """
    JSONL logger for repeatable navigation tests.

    A run contains:
      - events.jsonl: append-only event stream
      - summary.json: final outcome
      - frames/*.jpg: Gemini/diagnostic frames
    """

    def __init__(self, branch, objective, root_dir="navigation_runs"):
        self.branch = _safe_slug(branch)
        self.objective = objective
        self.root_dir = root_dir
        self.started_at = time.time()
        self.frame_count = 0

        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.run_id = "{}_{}".format(stamp, self.branch)
        self.run_dir = os.path.join(self.root_dir, self.run_id)
        self.frames_dir = os.path.join(self.run_dir, "frames")
        os.makedirs(self.frames_dir, exist_ok=True)

        self.events_path = os.path.join(self.run_dir, "events.jsonl")
        self.summary_path = os.path.join(self.run_dir, "summary.json")

        self.log_event("run_started", branch=self.branch, objective=objective)

    def _base_event(self, event_type):
        now = time.time()
        return {
            "ts": now,
            "elapsed_s": round(now - self.started_at, 3),
            "event_type": event_type,
            "run_id": self.run_id,
            "branch": self.branch,
        }

    def log_event(self, event_type, **payload):
        event = self._base_event(event_type)
        event.update(payload)
        with open(self.events_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
        return event

    def log_frame_b64(self, image_b64, label="frame", metadata=None):
        if not image_b64:
            return None

        self.frame_count += 1
        label = _safe_slug(label)
        filename = "{:04d}_{}.jpg".format(self.frame_count, label)
        path = os.path.join(self.frames_dir, filename)

        try:
            with open(path, "wb") as f:
                f.write(base64.b64decode(image_b64))
        except Exception as exc:
            self.log_event("frame_save_failed", label=label, error=str(exc))
            return None

        rel_path = os.path.relpath(path, self.run_dir)
        self.log_event("frame_saved", frame=rel_path, label=label, metadata=metadata or {})
        return rel_path

    def summarize(self, status, **metrics):
        summary = {
            "run_id": self.run_id,
            "branch": self.branch,
            "objective": self.objective,
            "status": status,
            "duration_s": round(time.time() - self.started_at, 3),
            "frames": self.frame_count,
        }
        summary.update(metrics)
        with open(self.summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2, sort_keys=True)
        self.log_event("run_finished", summary=summary)
        return summary
