#!/usr/bin/env python3
"""
memryx_yolo_posenet_count.py
- Single RTSP / video input
- Runs YOLO DFP on MemryX Simulator
- Optionally runs PoseNet DFP on person crops
- Maintains a simple centroid tracker to assign IDs and count unique people/vehicles

Prereqs:
  pip install opencv-python numpy memryx
  (memryx SDK installed per their instructions)
"""

import cv2
import numpy as np
import time
from memryx import Simulator  # simulator API
from collections import deque

# ---------- USER CONFIG ----------
YOLO_DFP = "yolo.dfp"           # compiled DFP with postprocessing -> boxes
POSENET_DFP = "posenet.dfp"     # optional: compiled posenet DFP
RTSP_URL = "rtsp://localhost:8554/stream1"  # or local file
FRAME_SKIP = 1                  # process every FRAME_SKIP-th frame
INPUT_SIZE = (640, 640)         # expected YOLO input (width, height)
CONF_TH = 0.4
IOU_TH = 0.45
USE_POSENET = False             # set True if you have posenet.dfp
# ------------------------------------------------

# --- simple non-max suppression (if needed) ---
def nms(boxes, scores, iou_threshold=0.45):
    if len(boxes) == 0:
        return []
    boxes = np.array(boxes)
    scores = np.array(scores)
    x1 = boxes[:,0]; y1 = boxes[:,1]; x2 = boxes[:,2]; y2 = boxes[:,3]
    areas = (x2-x1+1)*(y2-y1+1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w*h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    return keep

# --- simple centroid tracker ---
class CentroidTracker:
    def __init__(self, max_lost=30):
        self.next_object_id = 0
        self.objects = {}           # object_id -> centroid
        self.bboxes = {}            # object_id -> bbox
        self.lost = {}              # object_id -> lost frames
        self.max_lost = max_lost
        self.seen_ids = set()       # used for unique counting

    def update(self, detections):
        """
        detections: list of (x1,y1,x2,y2,score,class_id)
        returns dict object_id -> (bbox, class_id)
        """
        # compute centroids for detections
        centroids = []
        for (x1,y1,x2,y2,score,cid) in detections:
            cx = int((x1+x2)/2)
            cy = int((y1+y2)/2)
            centroids.append((cx,cy))

        if len(self.objects) == 0:
            # register all
            for i, c in enumerate(centroids):
                oid = self.next_object_id
                self.next_object_id += 1
                self.objects[oid] = c
                self.bboxes[oid] = detections[i][:4]
                self.lost[oid] = 0
                self.seen_ids.add((oid, detections[i][5]))  # track (id, class)
        else:
            # match by nearest centroid (greedy)
            o_ids = list(self.objects.keys())
            o_centroids = np.array([self.objects[i] for i in o_ids], dtype=np.float32)
            if len(centroids) == 0:
                # mark all as lost
                for oid in o_ids:
                    self.lost[oid] += 1
                # remove lost
                self._purge_lost()
                return self._export_current(detections)
            d_centroids = np.array(centroids, dtype=np.float32)
            # distances
            D = np.linalg.norm(o_centroids[:, None, :] - d_centroids[None, :, :], axis=2)
            # greedy match
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)
            assigned_dets = set()
            assigned_objs = set()
            # match
            for r in rows:
                c = cols[r]
                if r in assigned_objs or c in assigned_dets:
                    continue
                oid = o_ids[r]
                # assign
                self.objects[oid] = tuple(d_centroids[c].astype(int))
                self.bboxes[oid] = detections[c][:4]
                self.lost[oid] = 0
                assigned_dets.add(c)
                assigned_objs.add(r)
                self.seen_ids.add((oid, detections[c][5]))
            # unassigned detections -> new objects
            for idx, det in enumerate(detections):
                if idx not in assigned_dets:
                    oid = self.next_object_id
                    self.next_object_id += 1
                    self.objects[oid] = centroids[idx]
                    self.bboxes[oid] = det[:4]
                    self.lost[oid] = 0
                    self.seen_ids.add((oid, det[5]))
            # increase lost for unmatched objects
            for ridx, oid in enumerate(o_ids):
                if ridx not in assigned_objs:
                    self.lost[oid] += 1
            self._purge_lost()
        return self._export_current(detections)

    def _purge_lost(self):
        to_del = [oid for oid, l in self.lost.items() if l > self.max_lost]
        for oid in to_del:
            del self.objects[oid]; del self.bboxes[oid]; del self.lost[oid]

    def _export_current(self, detections):
        # return mapping id->(bbox, class)
        out = {}
        for oid in self.objects:
            out[oid] = (self.bboxes[oid], None)
        # to include class, we check seen_ids
        return out

    def unique_counts(self):
        # compute counts per class from seen_ids
        counts = {}
        for (oid, class_id) in self.seen_ids:
            counts[class_id] = counts.get(class_id, 0) + 1
        return counts

# ---------------- MemryX Simulator + main loop ----------------
def preprocess_frame(frame, input_w, input_h):
    # resize + normalize to float32 in [0,1], return shape [1,h,w,3]
    im = cv2.resize(frame, (input_w, input_h))
    arr = im.astype(np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

def run_demo():
    cap = cv2.VideoCapture(RTSP_URL)
    if not cap.isOpened():
        print("Failed to open stream/file:", RTSP_URL)
        return

    # create simulator instance for YOLO
    s_yolo = Simulator(dfp=YOLO_DFP)
    s_posenet = None
    if USE_POSENET:
        s_posenet = Simulator(dfp=POSENET_DFP)

    tracker = CentroidTracker(max_lost=60)
    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            if frame_idx % FRAME_SKIP != 0:
                continue

            h0, w0 = frame.shape[:2]
            inp = preprocess_frame(frame, INPUT_SIZE[0], INPUT_SIZE[1])  # [1,H,W,3]

            # call simulator: expects input shaped to the model's input_shape
            outputs = s_yolo.infer(inp, model_idx=0)  # outputs is list of np arrays
            # outputs shape & semantics depend on your DFP & postprocessing!
            # If your DFP included postprocessing, 'outputs' should be an array of detections per frame:
            # e.g., outputs[0] => array shape (N, 6) with [x1,y1,x2,y2,score,class]
            # Many MemryX tutorials compile YOLO with postprocessing so you get ready-to-use boxes.
            dets = []
            raw = outputs[0]  # shape e.g. (N,6) OR (N, num_outputs)
            # If output shape is (N,6) then parse directly:
            if raw.ndim == 2 and raw.shape[1] >= 6:
                for row in raw:
                    x1, y1, x2, y2, score, cls = row[:6]
                    if score < CONF_TH:
                        continue
                    # outputs are in normalized coords or model pixel coords depending on compile time;
                    # try both: if max coord <=1 assume normalized
                    if max(x1,y1,x2,y2) <= 1.01:
                        # normalized -> convert to original frame
                        x1 = int(x1 * w0); x2 = int(x2 * w0)
                        y1 = int(y1 * h0); y2 = int(y2 * h0)
                    else:
                        x1 = int(x1 * (w0/INPUT_SIZE[0]   if INPUT_SIZE[0] != 0 else 1))
                        x2 = int(x2 * (w0/INPUT_SIZE[0]   if INPUT_SIZE[0] != 0 else 1))
                        y1 = int(y1 * (h0/INPUT_SIZE[1]   if INPUT_SIZE[1] != 0 else 1))
                        y2 = int(y2 * (h0/INPUT_SIZE[1]   if INPUT_SIZE[1] != 0 else 1))
                    dets.append((x1,y1,x2,y2,float(score), int(cls)))
            else:
                # If outputs are raw feature maps, you must decode them with a YOLO decoder.
                # This is model-specific and error-prone; recommended: recompile DFP with post-processing.
                print("Output shape unexpected; please use a DFP compiled with postprocessing.")
                break

            # optional: run NMS just in case (if postprocessing didn't include NMS)
            if len(dets) > 0:
                boxes = [d[:4] for d in dets]
                scores = [d[4] for d in dets]
                keep = nms(boxes, scores, iou_threshold=IOU_TH)
                dets = [dets[i] for i in keep]

            # update tracker and counts
            tracker.update(dets)
            counts = tracker.unique_counts()

            # annotate frame
            for (x1,y1,x2,y2,score,cls) in dets:
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.putText(frame, f"{cls}:{score:.2f}", (x1, y1-6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

            # display counts
            y = 20
            for cls_id, c in counts.items():
                cv2.putText(frame, f"Class {cls_id} unique: {c}", (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
                y += 20

            cv2.imshow("out", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # PoseNet on person crops: call s_posenet.infer on cropped person images if enabled
            # ... (omitted for brevity; same pattern: crop, resize, preprocess, s_posenet.infer)

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run_demo()
