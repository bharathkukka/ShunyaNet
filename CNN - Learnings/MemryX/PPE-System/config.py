# config.py
CAMERA_ID = "cam1"
RTSP_URL = "rtsp://localhost:8554/stream1"
YOLO_DFP = "yolo_ppe.dfp"   # compiled to detect classes: person, hardhat, vest, vehicle, etc.
INPUT_SIZE = (640, 640)     # model input size
CONF_TH = 0.4
CLIP_PRE_SEC = 15           # seconds to include before event
CLIP_POST_SEC = 15          # seconds after event
FPS_ESTIMATE = 10           # used for writer frame rate; try to match stream
ALERT_COOLDOWN_SEC = 60     # per-camera cooldown between alerts
CLIPS_DIR = "./clips"
WEBHOOK_URL = "https://example.com/webhook"  # set to your alert receiver
SMTP_ENABLED = False
SMTP_CONFIG = {
    "host": "smtp.example.com",
    "port": 587,
    "user": "user@example.com",
    "password": "password",
    "to": ["safety@example.com"]
}
