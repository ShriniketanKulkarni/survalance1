# ─── detector.py ──────────────────────────────────────────────────────────────
"""
Thread-safe shared YOLO detector — one model instance shared across all cameras.
FP16 inference + cuDNN benchmark for maximum GPU throughput.

Weapon detection strategy (3 passes):
  Pass 1  — main model (yolo26m.pt), normal conf  → persons + bags only
  Pass 2  — best.pt, KNIFE_CONF  → knife detection  (best.pt class 1)
  Pass 3  — best.pt, FIREARM_CONF → firearm detection (best.pt class 0)

best.pt class map:  {0: 'guns', 1: 'knife'}
All weapon detection is handled exclusively by best.pt.
"""

import threading
import numpy as np
import torch
from ultralytics import YOLO
import config

# ── cuDNN performance flags ───────────────────────────────────────────────────
if config.DEVICE == "cuda":
    torch.backends.cudnn.benchmark    = True
    torch.backends.cudnn.deterministic = False

# ── Label maps ────────────────────────────────────────────────────────────────
KNIFE_LABELS = {
    1: "Knife",       # best.pt class 1
}

FIREARM_LABELS = {
    0: "Gun",         # best.pt class 0
}


class SharedDetector:
    """
    Singleton YOLO detector shared by all camera pipelines.
    A threading.Lock ensures only one inference runs at a time on the GPU.
    """
    _instance = None
    _lock      = threading.Lock()   # guards singleton creation only

    @classmethod
    def get(cls) -> "SharedDetector":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.device = config.DEVICE
        self.half   = (config.DEVICE == "cuda")
        self._infer_lock = threading.Lock()   # serialise concurrent inference

        # ── Main model ────────────────────────────────────────────────────────
        self.model = YOLO(config.YOLO_MODEL)
        if self.half:
            self.model.model.half()
            print(f"[Detector] Main={config.YOLO_MODEL}  Device=CUDA  FP16  (shared)")
        else:
            print(f"[Detector] Main={config.YOLO_MODEL}  Device=CPU   FP32  (shared)")

        # ── Optional dedicated weapon model ───────────────────────────────────
        self._wpn_model = None
        if config.WEAPON_DETECTION_MODEL:
            try:
                self._wpn_model = YOLO(config.WEAPON_DETECTION_MODEL)
                if self.half:
                    self._wpn_model.model.half()
                print(f"[Detector] WeaponModel={config.WEAPON_DETECTION_MODEL}  loaded ✓")
            except Exception as e:
                print(f"[Detector] WeaponModel load failed ({e}) — using main model for firearms")

        # ── GPU warm-up ───────────────────────────────────────────────────────
        try:
            dummy = np.zeros((480, 640, 3), dtype=np.uint8)
            self.model.predict(dummy, conf=0.9, device=self.device,
                               half=self.half, verbose=False)
            if self._wpn_model:
                self._wpn_model.predict(dummy, conf=0.9, device=self.device,
                                        half=self.half, verbose=False)
            print("[Detector] Warm-up complete.")
        except Exception as e:
            print(f"[Detector] Warm-up skipped: {e}")

    # ── Public inference API ──────────────────────────────────────────────────
    def detect(self, frame, conf_override: float | None = None):
        """
        Thread-safe inference on a BGR frame.

        Args:
            frame:         BGR numpy array.
            conf_override: If set, overrides config.YOLO_CONF for the main pass
                           (used to apply webcam-specific lower confidence).

        Returns:
            persons  (list of dicts) — {bbox, conf, cls}
            bags     (list of dicts) — {bbox, conf, cls}
            weapons  (list of dicts) — {bbox, conf, cls, label, weapon_type}
                                        weapon_type: "knife" | "firearm"
            misc_objects  (list of dicts) — {bbox, conf, cls, label}  (mouse, cell phone, …)
        """
        with self._infer_lock:
            with torch.no_grad():
                # ── Pass 1: main classes at normal conf ───────────────────────
                _conf = conf_override if conf_override is not None else config.YOLO_CONF
                res_main = self.model.predict(
                    frame,
                    conf=_conf,
                    iou=config.YOLO_IOU,
                    classes=config.YOLO_CLASSES,
                    imgsz=640,
                    device=self.device,
                    half=self.half,
                    verbose=False,
                )[0]

                # ── Pass 2: knife sweep via best.pt ───────────────────────────
                _wpn_src = self._wpn_model if self._wpn_model else self.model
                res_blade = _wpn_src.predict(
                    frame,
                    conf=config.KNIFE_CONF,
                    iou=config.YOLO_IOU,
                    classes=list(config.CLS_KNIFE_WEAPON),
                    imgsz=640,
                    device=self.device,
                    half=self.half,
                    verbose=False,
                )[0]

                # ── Pass 3: firearm sweep via best.pt ─────────────────────────
                res_fire = _wpn_src.predict(
                    frame,
                    conf=config.FIREARM_CONF,
                    iou=config.YOLO_IOU,
                    classes=list(config.CLS_FIREARMS),
                    imgsz=640,
                    device=self.device,
                    half=self.half,
                    verbose=False,
                )[0]

        persons, bags, weapons, misc_objects = [], [], [], []
        seen_weapon_boxes = set()   # coarse spatial dedup across all passes

        def _add_box(box, second_pass: bool = False, firearm_pass: bool = False):
            cls  = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            # Drop tiny spurious boxes
            if (x2 - x1) < 15 or (y2 - y1) < 15:
                return

            entry = {"bbox": [x1, y1, x2, y2], "conf": conf, "cls": cls}

            if firearm_pass:
                # Always treat as firearm regardless of class overlap
                key = ("fire", cls, x1 // 20, y1 // 20)
                if key not in seen_weapon_boxes:
                    seen_weapon_boxes.add(key)
                    entry["label"]       = FIREARM_LABELS.get(cls, "Firearm")
                    entry["weapon_type"] = "firearm"
                    weapons.append(entry)

            elif cls == config.CLS_PERSON:
                if not second_pass:
                    persons.append(entry)

            elif cls in config.CLS_BAGS:
                if not second_pass:
                    bags.append(entry)

            elif cls in config.CLS_MISC_OBJECTS:
                if not second_pass:
                    entry["label"] = config.MISC_OBJECT_NAMES.get(cls, "OBJECT")
                    misc_objects.append(entry)

            elif cls in config.CLS_KNIFE_WEAPON:
                key = ("blade", cls, x1 // 20, y1 // 20)
                if key not in seen_weapon_boxes:
                    seen_weapon_boxes.add(key)
                    entry["label"]       = KNIFE_LABELS.get(cls, "Knife")
                    entry["weapon_type"] = "knife"
                    weapons.append(entry)

        for box in res_main.boxes:
            _add_box(box, second_pass=False)
        for box in res_blade.boxes:
            _add_box(box, second_pass=True)
        for box in res_fire.boxes:
            _add_box(box, firearm_pass=True)

        if weapons:
            knives   = [w["label"] for w in weapons if w.get("weapon_type") == "knife"]
            firearms = [w["label"] for w in weapons if w.get("weapon_type") == "firearm"]
            if firearms:
                print(f"[Detector] 🔫 FIREARM detected: {firearms}")
            if knives:
                print(f"[Detector] 🔪 BLADE detected: {knives}")

        if misc_objects:
            labels = [o["label"] for o in misc_objects]
            print(f"[Detector] 📱 MISC OBJECTS detected: {labels}")

        return persons, bags, weapons, misc_objects


# Backwards-compat alias
Detector = SharedDetector
