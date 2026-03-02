# ring_buffer.py
import collections, time, os
import cv2
import numpy as np
from typing import Deque, Tuple, List
from config import CLIP_PRE_SEC, CLIP_POST_SEC, FPS_ESTIMATE, CLIPS_DIR, CAMERA_ID

class RingBuffer:
    def __init__(self, max_seconds=60, fps=FPS_ESTIMATE):
        self.max_frames = int(max_seconds * fps)
        self.fps = fps
        self.buffer: Deque[Tuple[float, np.ndarray]] = collections.deque(maxlen=self.max_frames)

    def push(self, frame: np.ndarray):
        self.buffer.append((time.time(), frame.copy()))

    def get_clip(self, event_time: float, pre_sec=CLIP_PRE_SEC, post_sec=CLIP_POST_SEC) -> List[Tuple[float, np.ndarray]]:
        start_t = event_time - pre_sec
        end_t = event_time + post_sec
        selected = [ (ts, fr) for (ts, fr) in self.buffer if ts >= start_t and ts <= end_t ]
        return selected

    def save_clip(self, event_time: float, filepath: str, pre_sec=CLIP_PRE_SEC, post_sec=CLIP_POST_SEC):
        frames = self.get_clip(event_time, pre_sec, post_sec)
        if not frames:
            print("[ringbuffer] no frames for clip")
            return False
        # Save using OpenCV VideoWriter
        first_frame = frames[0][1]
        h, w = first_frame.shape[:2]
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # mp4
        writer = cv2.VideoWriter(filepath, fourcc, self.fps, (w,h))
        for ts, fr in frames:
            writer.write(fr)
        writer.release()
        return True
