# pose_logic.py
import math
from typing import List, Dict, Tuple
from collections import deque
from config_pose import KP_CONF_TH, FALL_ANGLE_THRESHOLD, BENDING_ANGLE_THRESHOLD, SITTING_RATIO, RUNNING_SPEED_THRESHOLD, IDLE_TIME_SEC

# human keypoint indices (COCO style)
# 0: nose, 1:left eye, 2:right eye, 3:left ear, 4:right ear,
# 5:left shoulder,6:right shoulder,7:left elbow,8:right elbow,9:left wrist,10:right wrist,
# 11:left hip,12:right hip,13:left knee,14:right knee,15:left ankle,16:right ankle
L_SHOULDER = 5; R_SHOULDER = 6; L_HIP = 11; R_HIP = 12; L_KNEE=13; R_KNEE=14; NOSE=0

def get_kp(coords, idx):
    x,y,c = coords[idx]
    return (x,y,c)

def angle_between(p1, p2, p3):
    """Angle at p2 formed by p1-p2-p3 in degrees"""
    ax,ay = p1; bx,by = p2; cx,cy = p3
    v1 = (ax-bx, ay-by); v2 = (cx-bx, cy-by)
    dot = v1[0]*v2[0] + v1[1]*v2[1]
    n1 = math.hypot(*v1); n2 = math.hypot(*v2)
    if n1==0 or n2==0:
        return 0.0
    cosang = max(-1.0, min(1.0, dot / (n1*n2)))
    return math.degrees(math.acos(cosang))

def torso_tilt_deg(kps):
    # use shoulders and hips to compute torso vector
    ls = get_kp(kps, L_SHOULDER); rs = get_kp(kps, R_SHOULDER)
    lh = get_kp(kps, L_HIP); rh = get_kp(kps, R_HIP)
    # take midpoints
    sx = (ls[0]+rs[0])/2; sy = (ls[1]+rs[1])/2
    hx = (lh[0]+rh[0])/2; hy = (lh[1]+rh[1])/2
    # vector from hip->shoulder
    vx = sx - hx; vy = sy - hy
    # vertical vector (0,-1) in image coords y downwards: use vector (0,-1)
    # angle between torso vector and vertical:
    # compute angle between (vx,vy) and (0,-1)
    dot = vx*0 + vy*(-1)
    n1 = math.hypot(vx,vy); n2 = 1.0
    if n1 == 0: return 0.0
    cos = max(-1.0, min(1.0, dot / n1))
    ang = math.degrees(math.acos(cos))
    return ang

def knee_angle(kps):
    # compute mean knee angle (hip-knee-ankle) for both legs
    angs = []
    for hip_i, knee_i, ank_i in [(L_HIP, L_KNEE, 15), (R_HIP, R_KNEE, 16)]:
        hip = get_kp(kps, hip_i); knee = get_kp(kps, knee_i); ank = get_kp(kps, ank_i)
        if hip[2] < KP_CONF_TH or knee[2] < KP_CONF_TH or ank[2] < KP_CONF_TH:
            continue
        ang = angle_between((hip[0],hip[1]), (knee[0],knee[1]), (ank[0],ank[1]))
        angs.append(ang)
    if not angs:
        return None
    return sum(angs)/len(angs)

def classify_pose(kps, bbox_norm, speed, history_times):
    """
    kps: list[(x_norm,y_norm,conf)...]
    bbox_norm: (x1,y1,x2,y2) normalized
    speed: centroid speed (normed)
    history_times: list of (t,centroid) for idle detection
    returns label string and reason dict
    """
    # basic checks
    if kps is None or len(kps) < 5:
        return "unknown", {"reason":"insufficient_kps"}

    # torso tilt
    ttilt = torso_tilt_deg(kps)  # degrees from vertical (0 = upright)
    knee_ang = knee_angle(kps)
    x1,y1,x2,y2 = bbox_norm
    height = y2 - y1
    # sitting heuristic: knee angle small (<120 deg) or bbox height relatively small
    sitting = False
    if knee_ang is not None and knee_ang < 120:
        sitting = True
    elif height < 0.4:
        sitting = True

    bending = False
    if ttilt > BENDING_ANGLE_THRESHOLD and knee_ang and knee_ang > 90:
        # torso bent forward with knees not folded
        bending = True

    # fall: large torso tilt AND sudden vertical displacement or high tilt + low motion (on ground)
    fall = False
    if ttilt > FALL_ANGLE_THRESHOLD:
        # check recent speed: if centroid rapidly moved or y increased beyond threshold
        if speed > 0.3 or ttilt > (FALL_ANGLE_THRESHOLD*1.3):
            fall = True

    running = False
    if speed > RUNNING_SPEED_THRESHOLD:
        running = True

    # idle detection: if centroid history shows little movement for IDLE_TIME_SEC
    idle = False
    if len(history_times) > 0:
        # compute displacement over last IDLE_TIME_SEC
        now = history_times[-1][0]
        start_time = now - IDLE_TIME_SEC
        pts = [p for (t,p) in history_times if t >= start_time]
        if len(pts) >= 2:
            xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
            dx = max(xs) - min(xs); dy = max(ys) - min(ys)
            if dx < 0.02 and dy < 0.02:  # almost stationary, tiny threshold
                idle = True

    # working: heuristic: hands moving, not idle, upright, within bbox -- detect if wrist kp exists and distance from shoulder small & wrist moving
    # Simplified: if not idle, not sitting, and speed < running threshold => 'working' (active but not running)
    working = False
    if not sitting and not running and not idle and not fall and not bending:
        working = True

    # Decide label with priority: fall > running > bending > sitting > idle > working
    label = "unknown"
    if fall:
        label = "fall"
    elif running:
        label = "running"
    elif bending:
        label = "bending"
    elif sitting:
        label = "sitting"
    elif idle:
        label = "idle"
    elif working:
        label = "working"
    else:
        label = "unknown"

    reason = {"torso_tilt": ttilt, "knee_angle": knee_ang, "height": height, "speed": speed,
              "sitting": sitting, "bending": bending, "fall": fall, "running": running, "idle": idle}
    return label, reason
