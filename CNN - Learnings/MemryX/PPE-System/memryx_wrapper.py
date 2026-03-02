# memryx_wrapper.py
import numpy as np
import cv2
import time
from typing import List, Tuple

# If memryx SDK is available:
try:
    from memryx import Simulator, AsyncAccl
    MEMRYX_AVAILABLE = True
except Exception:
    MEMRYX_AVAILABLE = False

class MemryxYolo:
    def __init__(self, dfp_path: str, input_size=(640,640), use_simulator=True):
        self.dfp = dfp_path
        self.input_w, self.input_h = input_size
        self.use_sim = use_simulator or not MEMRYX_AVAILABLE
        if self.use_sim:
            if not MEMRYX_AVAILABLE:
                raise RuntimeError("Memryx package not found; please install SDK or set use_simulator=False")
            self.sess = Simulator(dfp=self.dfp)
        else:
            # Example: AsyncAccl usage (requires hardware + proper init)
            self.sess = AsyncAccl(dfp=self.dfp)
            self.sess.start()

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        im = cv2.resize(frame, (self.input_w, self.input_h))
        arr = im.astype(np.float32) / 255.0
        arr = np.expand_dims(arr, axis=0)  # shape [1, H, W, 3]
        return arr

    def infer(self, frame: np.ndarray):
        """
        Returns detections list: each detection is (x1,y1,x2,y2,score,class_id)
        NOTE: assumes your DFP produces postprocessed boxes. If your DFP returns raw
              tensors you must decode them appropriately or recompile with postprocessing.
        """
        inp = self.preprocess(frame)
        out = self.sess.infer(inp, model_idx=0)  # returns list of arrays in memryx SDK
        # typical: out[0] shape (N,6) -> x1,y1,x2,y2,score,class
        if len(out) == 0:
            return []
        arr = out[0]
        # convert normalized coords if needed will be handled upstream
        return arr.tolist()
