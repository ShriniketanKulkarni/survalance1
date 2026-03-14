# ─── config.py ────────────────────────────────────────────────────────────────
# Central configuration — Smart Surveillance System
import torch
# ── YOLO ──────────────────────────────────────────────────────────────────────
YOLO_MODEL   = "yolo26m.pt"         # auto-downloaded on first run
YOLO_CONF    = 0.20              # lower threshold for better CCTV recall
YOLO_IOU     = 0.40              # tighter IOU for cleaner overlapping boxes

# ── Webcam-specific accuracy settings ────────────────────────────────────────
# Only applied when the video source is an integer (live webcam).
WEBCAM_CONF            = 0.17    # slightly lower conf for live/indoor conditions
WEBCAM_DETECT_EVERY_N  = 1      # run YOLO on every frame (no stale detections)
VIDEO_DETECT_EVERY_N   = 2      # every 2nd frame is fine for video files
WEBCAM_WARMUP_FRAMES   = 25     # skip first N frames (auto-exposure settling)
CLAHE_ENABLED          = True   # CLAHE contrast enhancement for webcam frames
if torch.cuda.is_available():
    DEVICE       = "cuda"
else:
    DEVICE       = "cpu"               # "cpu" if no GPU

# COCO classes to detect
CLS_PERSON   = 0
CLS_BAGS     = {24, 26, 28}         # backpack, handbag, suitcase
# Miscellaneous objects of interest detected by the main model
CLS_MISC_OBJECTS  = {64, 67}        # mouse(64), cell phone(67)
MISC_OBJECT_NAMES = {64: "MOUSE", 67: "CELL PHONE"}

# Sharp-object / blade weapons  (COCO-based model — kept for reference only)
CLS_WEAPONS  = {43, 76}             # knife(43), scissors(76-proxy) — NOT used; best.pt handles knives

# best.pt knife class (used by the weapon model — best.pt class 1 = knife)
CLS_KNIFE_WEAPON = {1}              # knife class in best.pt

# ── Firearm classes ────────────────────────────────────────────────────────────
# Standard COCO has NO firearm class.  Switch to a weapon-specialised model
# (e.g. Roboflow "Weapon Detection" or Ultralytics Hub) that assigns:
#   0 = Pistol / Handgun
#   1 = Rifle / Assault Rifle
#   2 = Handgun (some models split pistol/handgun)
# If your model uses different IDs, update this set accordingly.
CLS_FIREARMS = {0}                  # guns(0) in best.pt
FIREARM_CONF = 0.60                # confidence threshold for firearms
KNIFE_CONF   = 0.11             # confidence threshold for knives (best.pt produces 0.15–0.50 range)

# All classes passed to the main inference pass
# Note: if YOLO_MODEL is a weapon-specialised model, classes 0-2 will map to
# firearms above and CLS_PERSON must be remapped (set CLS_PERSON to the
# person-class ID of that model, usually 3 or 4).
YOLO_CLASSES = [0, 24, 26, 28, 64, 67]  # person(0), bags(24,26,28), mouse(64), cell phone(67) — knives/firearms handled by best.pt
WEAPON_DETECTION_MODEL = "best.pt"  # dedicated weapon model — handles both knives & firearms

# ── TRACKER ───────────────────────────────────────────────────────────────────
MAX_AGE      = 40              # keep track alive longer for re-identification
N_INIT       = 2               # confirm track after 2 frames

# ── BEHAVIOUR THRESHOLDS ─────────────────────────────────────────────────────
LOITER_SECONDS      = 10            # seconds stationary → loitter alert
LOITER_RADIUS_PX    = 80            # movement within this radius = "stationary"
ABANDON_SECONDS     = 8             # seconds bag alone → abandoned alert
ABANDON_RADIUS_PX   = 120           # proximity to count person as "near" bag
RUNNING_PX_PER_SEC  = 80            # pixel/sec speed above which = "running" (lower = more sensitive)

# ── RESTRICTED ZONES ─────────────────────────────────────────────────────────
# List of polygons — each polygon is a list of [x, y] in pixel coords.
# Updated live from the Flask UI zone-editor.
RESTRICTED_ZONES = []

# ── FIREARMS ZONE ────────────────────────────────────────────────────────────
# Polygon zone where firearm presence triggers an immediate priority alert.
FIREARMS_ZONE = []

# ── SNAPSHOTS ────────────────────────────────────────────────────────────────
SNAPSHOT_DIR    = "snapshots"       # folder to save alert snapshots
SAVE_SNAPSHOTS  = True

# ── VIDEO SOURCE ──────────────────────────────────────────────────────────────
DEFAULT_SOURCE  = "test_video.mp4"

# ── ALERT SOUND ──────────────────────────────────────────────────────────────
ALERT_SOUND     = True

# ── EMAIL ALERTS ──────────────────────────────────────────────────────────────
EMAIL_ENABLED       = False           # set True after configuring credentials
EMAIL_SMTP_HOST     = "smtp.gmail.com"
EMAIL_SMTP_PORT     = 587             # 587 = STARTTLS,  465 = SSL
EMAIL_SENDER        = ""              # your Gmail / sender address
EMAIL_PASSWORD      = ""              # app-password (not your login password)
EMAIL_RECIPIENTS    = []              # list of recipient addresses
# Minimum seconds between emails for the SAME alert type (prevents spam)
EMAIL_COOLDOWN_SEC  = 60
# Which severities trigger an email (CRITICAL | HIGH | MEDIUM | LOW)
EMAIL_MIN_SEVERITY  = "HIGH"          # send for HIGH and CRITICAL

# ── FLASK ─────────────────────────────────────────────────────────────────────
FLASK_HOST      = "0.0.0.0"
FLASK_PORT      = 5000
STREAM_FPS      = 25
JPEG_QUALITY    = 85
