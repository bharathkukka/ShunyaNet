# tracker.py
import numpy as np
import time

class SimpleTracker:
    def __init__(self, max_lost=30):
        self.next_id = 0
        self.objects = {}  # id -> centroid (x_norm,y_norm)
        self.bboxes = {}   # id -> last bbox (x1,y1,x2,y2) normalized
        self.lost = {}
        self.history = {}  # id -> deque of (t, centroid)
        self.max_lost = max_lost

    def update(self, detections):  # detections: list of (centroid_x, centroid_y, bbox_norm)
        # detections: each (cx,cy,(x1,y1,x2,y2))
        if len(self.objects) == 0:
            for det in detections:
                cid = self.next_id; self.next_id += 1
                self.objects[cid] = (det[0], det[1])
                self.bboxes[cid] = det[2]
                self.lost[cid] = 0
                self.history[cid] = [(time.time(), det[0], det[1])]
            return self.objects.keys()

        # match by nearest centroid
        o_ids = list(self.objects.keys())
        o_c = np.array([self.objects[i] for i in o_ids])
        d_c = np.array([[d[0], d[1]] for d in detections]) if detections else np.array([])
        assigned = set()
        if d_c.size > 0:
            D = np.linalg.norm(o_c[:,None,:] - d_c[None,:,:], axis=2)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)
            assigned_det = set()
            for r in rows:
                c = cols[r]
                if r in assigned or c in assigned_det:
                    continue
                oid = o_ids[r]
                self.objects[oid] = (float(d_c[c][0]), float(d_c[c][1]))
                self.bboxes[oid] = detections[c][2]
                self.lost[oid] = 0
                self.history[oid].append((time.time(), float(d_c[c][0]), float(d_c[c][1])))
                # keep history short
                if len(self.history[oid]) > 20:
                    self.history[oid].pop(0)
                assigned.add(r); assigned_det.add(c)
            # new detections
            for i, det in enumerate(detections):
                if i not in assigned_det:
                    nid = self.next_id; self.next_id += 1
                    self.objects[nid] = (det[0], det[1])
                    self.bboxes[nid] = det[2]
                    self.lost[nid] = 0
                    self.history[nid] = [(time.time(), det[0], det[1])]
            # increment lost for unmatched objects
            for ridx, oid in enumerate(o_ids):
                if ridx not in assigned:
                    self.lost[oid] += 1
            # purge lost
            to_del = [oid for oid, l in self.lost.items() if l > self.max_lost]
            for oid in to_del:
                del self.objects[oid]; del self.bboxes[oid]; del self.lost[oid]; del self.history[oid]
        else:
            # no detections: mark all as lost
            for oid in o_ids:
                self.lost[oid] += 1
            to_del = [oid for oid, l in self.lost.items() if l > self.max_lost]
            for oid in to_del:
                del self.objects[oid]; del self.bboxes[oid]; del self.lost[oid]; del self.history[oid]

    def get_speed(self, oid):
        """Return approximate speed (normed per-frame) using history"""
        hist = self.history.get(oid, [])
        if len(hist) < 2:
            return 0.0
        # compute average speed between last two entries
        (t1, x1, y1), (t2, x2, y2) = hist[-2], hist[-1]
        dt = t2 - t1
        if dt <= 0:
            return 0.0
        dist = ((x2-x1)**2 + (y2-y1)**2)**0.5
        return dist / dt
