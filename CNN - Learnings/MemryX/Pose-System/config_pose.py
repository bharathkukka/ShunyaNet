# config_pose.py
CAMERA_ID = "cam1"
RTSP_URL = "rtsp://localhost:8554/stream1"
POSENET_DFP = "posenet.dfp"
INPUT_SIZE = (256, 256)   # model input expected by your PoseNet (w,h)
KP_CONF_TH = 0.3         # min keypoint confidence to consider
CLIP_PRE_SEC = 15
CLIP_POST_SEC = 15
FPS_EST = 10
ALERT_COOLDOWN = 60      # seconds between repeated alerts per camera
# Activity thresholds (tunable)
FALL_ANGLE_THRESHOLD = 45.0    # degrees from vertical torso tilt to consider fall-like
FALL_VEL_THRESHOLD = 0.5       # normalized position change per frame (tunable)
BENDING_ANGLE_THRESHOLD = 60.0 # hip angle below this = bending
SITTING_RATIO = 0.5            # bbox height ratio or knee angle threshold
RUNNING_SPEED_THRESHOLD = 0.02 # normalized centroid speed (per-frame) for running
IDLE_TIME_SEC = 30             # if person remains near same spot with low motion -> idle
