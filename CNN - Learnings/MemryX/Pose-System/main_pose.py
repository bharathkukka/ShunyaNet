# main_pose.py
import cv2, time, os
from config_pose import CAMERA_ID, RTSP_URL, CLIP_PRE_SEC, CLIP_POST_SEC, FPS_EST
from memryx_posenet import MemryxPose
from ring_buffer import RingBuffer
from tracker import SimpleTracker
from pose_logic import classify_pose
from alerting import raise_alert

POSENET_DFP = "posenet.dfp"
USE_SIM = True

def ensure_dir(d): os.makedirs(d, exist_ok=True); return d

def extract_centroid_and_bbox_from_kps(kps):
    # compute bbox from visible keypoints and centroid
    xs = [kp[0] for kp in kps if kp[2] >= 0.2]
    ys = [kp[1] for kp in kps if kp[2] >= 0.2]
    if not xs or not ys:
        return None, None
    x1 = min(xs); x2 = max(xs); y1 = min(ys); y2 = max(ys)
    cx = (x1+x2)/2; cy = (y1+y2)/2
    return (cx,cy), (x1,y1,x2,y2)

def main():
    pose = MemryxPose(dfp_path=POSENET_DFP, input_size=(256,256), use_sim=USE_SIM)
    rb = RingBuffer(max_seconds=CLIP_PRE_SEC+CLIP_POST_SEC+10, fps=FPS_EST)
    tracker = SimpleTracker(max_lost=60)
    last_alert = 0

    cap = cv2.VideoCapture(RTSP_URL)
    if not cap.isOpened():
        print("failed open", RTSP_URL); return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1); continue
            tnow = time.time()
            rb.push(frame)
            # run pose net on full frame (or on person crops if you run detection first)
            persons_kps = pose.infer(frame)  # list of persons -> list of (x_norm,y_norm,conf)
            dets = []
            for kps in persons_kps:
                centroid, bbox = extract_centroid_and_bbox_from_kps(kps)
                if centroid is None: continue
                dets.append((centroid[0], centroid[1], bbox))
            # update tracker
            tracker.update(dets)
            # for each tracked id compute speed and classify
            for oid in list(tracker.objects.keys()):
                centroid = tracker.objects[oid]
                bbox = tracker.bboxes[oid]
                speed = tracker.get_speed(oid)
                history = tracker.history.get(oid, [])
                # also we need the latest keypoints for that object: map by nearest centroid
                # find closest person_kps by centroid
                matched_kps = None
                min_d = 1e9
                for kps in persons_kps:
                    # compute centroid for kps
                    xs = [kp[0] for kp in kps if kp[2]>=0.2]; ys = [kp[1] for kp in kps if kp[2]>=0.2]
                    if not xs: continue
                    cx = sum(xs)/len(xs); cy = sum(ys)/len(ys)
                    d = ((cx-centroid[0])**2 + (cy-centroid[1])**2)**0.5
                    if d < min_d:
                        min_d = d; matched_kps = kps
                label, reason = classify_pose(matched_kps, bbox, speed, history)
                # annotate frame for visualization
                x1,y1,x2,y2 = [int(v*frame.shape[1]) if i%2==0 else int(v*frame.shape[0]) for i,v in enumerate(bbox*2)] if False else None
                # simplified draw: skip if bbox None
                # handle alerts
                if label in ("fall", "running") or (label=="idle" and len(history)>0 and time.time()-history[0][0] > 60):
                    # throttle per-camera
                    if time.time() - last_alert >  ALRT_COOLDOWN:
                        last_alert = time.time()
                        ts = time.strftime("%Y%m%d_%H%M%S")
                        clip_dir = ensure_dir(os.path.join("clips", CAMERA_ID))
                        clip_path = os.path.join(clip_dir, f"pose_alert_{label}_{ts}.mp4")
                        saved = rb.save_clip(time.time(), clip_path)
                        details = {"label": label, "reason": reason, "tracker_id": oid}
                        ok = raise_alert(CAMERA_ID, clip_path, details)
                        print("Alert:", label, ok, "saved:", saved)
            # display optionally
            cv2.imshow("pose", frame)
            if cv2.waitKey(1)&0xFF==ord('q'): break

    finally:
        cap.release(); cv2.destroyAllWindows()
