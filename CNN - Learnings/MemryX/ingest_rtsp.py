# ingest_rtsp.py
# script (ingest_rtsp.py) was designed for background AI processing.
# It used two threads for every stream (one to read, one to process).
import cv2
import threading
import queue
import time

RTSP_URLS = [f"rtsp://localhost:8554/stream{i}" for i in range(1, 101)]
FPS_TARGET = 5  # desired processing fps per stream (you can reduce)
QUEUE_MAX = 2

def rtsp_worker(url, frame_queue):
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)  # use FFMPEG backend if available
    if not cap.isOpened():
        print("Failed to open", url)
        return
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue
        # optionally resize to reduce processing cost
        # frame = cv2.resize(frame, (640, 480))
        try:
            frame_queue.put_nowait(frame)
        except queue.Full:
            # drop the oldest frame then put (simple drop strategy)
            try:
                _ = frame_queue.get_nowait()
                frame_queue.put_nowait(frame)
            except queue.Full:
                pass
        # throttle to target FPS to avoid hogging CPU
        time.sleep(1.0 / FPS_TARGET)

def inference_worker(url, frame_queue, worker_id):
    while True:
        frame = frame_queue.get()
        # Here: send frame to MemryX inference API / model
        # e.g., memryx.run_inference(frame) -- placeholder
        # (replace with actual MemryX SDK calls / compiled model)
        # For now, just mark timestamp
        print(f"[{worker_id}] processed frame from {url} at {time.time()}")
        frame_queue.task_done()

def main():
    frame_queues = {}
    for i, url in enumerate(RTSP_URLS):
        q = queue.Queue(maxsize=QUEUE_MAX)
        frame_queues[url] = q
        threading.Thread(target=rtsp_worker, args=(url, q), daemon=True).start()
        threading.Thread(target=inference_worker, args=(url, q, i+1), daemon=True).start()

    # keep main alive
    while True:
        time.sleep(60)

if __name__ == "__main__":
    main()
