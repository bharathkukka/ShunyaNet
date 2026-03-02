# memryx_posenet.py
import numpy as np, cv2
from typing import List
from config_pose import INPUT_SIZE
try:
    from memryx import Simulator, AsyncAccl
    MEMRYX_AVAILABLE = True
except Exception:
    MEMRYX_AVAILABLE = False

class MemryxPose:
    def __init__(self, dfp_path: str, input_size=INPUT_SIZE, use_sim=True):
        self.dfp = dfp_path
        self.w, self.h = input_size
        self.use_sim = use_sim or not MEMRYX_AVAILABLE
        if self.use_sim:
            if not MEMRYX_AVAILABLE:
                raise RuntimeError("memryx package not installed; install SDK or set use_sim=False")
            self.sess = Simulator(dfp=self.dfp)
        else:
            self.sess = AsyncAccl(dfp=self.dfp)
            self.sess.start()

    def preprocess(self, frame):
        im = cv2.resize(frame, (self.w, self.h))
        arr = im.astype(np.float32) / 255.0
        arr = np.expand_dims(arr, axis=0)  # [1,H,W,3]
        return arr

    def infer(self, frame):
        """
        Returns: list of persons where each person = list of keypoints: [(x_norm,y_norm,conf), ...]
        x_norm,y_norm in [0,1] relative to frame width/height.
        The exact format depends on your compiled DFP — adapt parsing accordingly.
        """
        inp = self.preprocess(frame)
        out = self.sess.infer(inp, model_idx=0)
        if len(out) == 0:
            return []
        # Example: out[0] shape (N, K, 3) OR (N, K*3)
        raw = out[0]
        # Normalize / parse:
        # Try common shapes
        if raw.ndim == 3:
            # (N, K, 3)
            persons = []
            for p in raw:
                kps = []
                for kp in p:
                    x, y, c = float(kp[0]), float(kp[1]), float(kp[2])
                    kps.append((x, y, c))
                persons.append(kps)
            return persons
        elif raw.ndim == 2:
            # (N, K*3) flatten
            N, M = raw.shape
            K = M // 3
            persons = []
            for i in range(N):
                row = raw[i]
                kps = []
                for k in range(K):
                    x = float(row[3*k+0]); y = float(row[3*k+1]); c = float(row[3*k+2])
                    kps.append((x,y,c))
                persons.append(kps)
            return persons
        else:
            # Unexpected: user should adapt based on compiled DFP
            return []
