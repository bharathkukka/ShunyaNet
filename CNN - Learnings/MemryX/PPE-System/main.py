# main_pose.py
import time, os
import cv2
from config import RTSP_URL, CAMERA_ID, CLIPS_DIR, ALERT_COOLDOWN_SEC, FPS_ESTIMATE
from memryx_wrapper import MemryxYolo
from ring_buffer import RingBuffer
from ppe_logic import group_detections, any_violation
from alerting import raise_alert

# Config
YOLO_DFP = "yolo_ppe.dfp"
INPUT_SIZE = (640,640)
USE_SIMULATOR = True

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)
    return d

def main():
    # Initialize components
    yolo = MemryxYolo(dfp_path=YOLO_DFP, input_size=INPUT_SIZE, use_simulator=USE_SIMULATOR)
    rb = RingBuffer(max_seconds=120, fps=FPS_ESTIMATE)  # keep 2 minutes buffer
    last_alert_time = 0

    cap = cv2.VideoCapture(RTSP_URL)
    if not cap.isOpened():
        print("Failed to open", RTSP_URL)
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue
            t_now = time.time()
            rb.push(frame)

            # Inference (you might want to skip frames to reduce load)
            dets = yolo.infer(frame)  # List rows [x1,y1,x2,y2,score,class]
            # If your model returns normalized coords, convert to pixel coords here.
            # For demo assume outputs are in pixel coords.

            p_results = group_detections(dets)
            if any_violation(p_results):
                # throttle alerts
                if t_now - last_alert_time > ALERT_COOLDOWN_SEC:
                    last_alert_time = t_now
                    # save clip
                    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(t_now))
                    clip_dir = ensure_dir(os.path.join(CLIPS_DIR, CAMERA_ID))
                    clip_path = os.path.join(clip_dir, f"violation_{timestamp}.mp4")
                    saved = rb.save_clip(t_now, clip_path)
                    # prepare details
                    details = {"persons": p_results}
                    ok = raise_alert(CAMERA_ID, clip_path, details)
                    print("Alert raised:", ok, "clip saved:", saved)
            # optional: display small UI with bounding boxes
            for r in p_results:
                x1,y1,x2,y2 = map(int, r['person_bbox'])
                color = (0,255,0) if r['compliant'] else (0,0,255)
                cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
                txt = "OK" if r['compliant'] else "NO PPE"
                cv2.putText(frame, txt, (x1, y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.imshow("PPE", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
