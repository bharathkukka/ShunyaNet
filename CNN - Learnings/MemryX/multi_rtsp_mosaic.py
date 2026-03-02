#!/usr/bin/env python3
"""
multi_rtsp_mosaic.py

Decode multiple RTSP streams and show n feeds on-screen in a grid mosaic.

Usage:
  - Edit RTSP_URLS list below or generate programmatically.
  - Run: python3 multi_rtsp_mosaic.py
  - Press 'q' to quit.

script (multi_rtsp_mosaic.py) is designed for visual display.
 It has a different, more efficient architecture for this task:

N Reader Threads: It creates one thread per stream (rtsp_reader_worker).

One Main Thread: The main() function runs in the single main thread.
"""

import cv2
import threading
import queue
import time
import numpy as np
import math

# ---------- USER CONFIG ----------
# Replace RTSP URLs or generate e.g. rtsp://localhost:8554/stream1...
RTSP_URLS = [f"rtsp://localhost:8554/stream{i}" for i in range(1, 13)]  # example: 12 streams
TARGET_THUMBNAIL_SIZE = (320, 180)   # width, height of each tile in mosaic
TARGET_FPS = 6                        # target per-stream polling interval -> 1/TARGET_FPS sec
QUEUE_MAX = 1                         # keep only the latest frame per stream
WINDOW_NAME = "RTSP Mosaic"
# ----------------------------------

def open_capture(url):
    """
    Return a cv2.VideoCapture capable of opening the RTSP url.
    You can customize this to use explicit GStreamer pipeline strings on platforms with hw decode.
    """
    # Try with default (FFMPEG) backend
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    return cap

def rtsp_reader_worker(url, frame_slot, stop_event, target_fps=TARGET_FPS):
    cap = open_capture(url)
    if not cap.isOpened():
        print(f"[reader] WARNING: failed to open {url}")
        # keep trying to reconnect until stop_event set
        while not stop_event.is_set():
            time.sleep(2.0)
            cap = open_capture(url)
            if cap.isOpened():
                print(f"[reader] reconnected {url}")
                break
        if not cap.isOpened():
            return

    read_interval = 1.0 / target_fps if target_fps > 0 else 0
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret or frame is None:
            # failed to read; sleep briefly and try again
            time.sleep(0.2)
            continue
        # put newest frame into the single-slot container (non-blocking)
        try:
            # empty existing frame if present
            try:
                frame_slot.get_nowait()
            except queue.Empty:
                pass
            frame_slot.put_nowait(frame)
        except queue.Full:
            pass
        if read_interval > 0:
            time.sleep(read_interval)
    try:
        cap.release()
    except Exception:
        pass
    print(f"[reader] stopped {url}")

def compute_grid(n):
    """
    Compute rows, cols for n tiles trying to be as square as possible.
    """
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    return rows, cols

def build_mosaic(frames, thumb_w, thumb_h, rows, cols):
    """
    frames: list of frames or None of length <= rows*cols
    returns: mosaic image (BGR)
    """
    tile_w, tile_h = thumb_w, thumb_h
    mosaic_h = rows * tile_h
    mosaic_w = cols * tile_w
    mosaic = np.zeros((mosaic_h, mosaic_w, 3), dtype=np.uint8)

    for idx in range(rows * cols):
        r = idx // cols
        c = idx % cols
        x = c * tile_w
        y = r * tile_h
        if idx < len(frames) and frames[idx] is not None:
            f = frames[idx]
            # resize keeping aspect ratio center-cropped if needed
            h, w = f.shape[:2]
            # simple resize ignoring crop (faster). Optionally add letterbox/crop logic.
            thumb = cv2.resize(f, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
            mosaic[y:y+tile_h, x:x+tile_w] = thumb
        else:
            # leave black tile or put a placeholder text
            cv2.rectangle(mosaic, (x, y), (x+tile_w-1, y+tile_h-1), (50,50,50), -1)
            cv2.putText(mosaic, "No feed", (x+10, y+tile_h//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1, cv2.LINE_AA)
    return mosaic

def main():
    n_streams = len(RTSP_URLS)
    rows, cols = compute_grid(n_streams)
    thumb_w, thumb_h = TARGET_THUMBNAIL_SIZE

    # per-stream single-slot queue to keep latest frame
    frame_slots = [queue.Queue(maxsize=QUEUE_MAX) for _ in range(n_streams)]
    stop_event = threading.Event()
    threads = []

    # start reader threads
    for i, url in enumerate(RTSP_URLS):
        t = threading.Thread(target=rtsp_reader_worker,
                             args=(url, frame_slots[i], stop_event),
                             daemon=True)
        t.start()
        threads.append(t)
        time.sleep(0.02)  # slight stagger reduce spikes

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    # optionally set window size to show grid at natural size
    cv2.resizeWindow(WINDOW_NAME, cols * thumb_w, rows * thumb_h)

    try:
        while True:
            # collect latest frames (non-blocking)
            frames = []
            for q in frame_slots:
                try:
                    f = q.get_nowait()
                    frames.append(f)
                except queue.Empty:
                    frames.append(None)

            # build mosaic from collected frames
            mosaic = build_mosaic(frames, thumb_w, thumb_h, rows, cols)

            # show overlay: timestamps and feed count
            info = f"Feeds: {n_streams} | {time.strftime('%Y-%m-%d %H:%M:%S')}"
            cv2.putText(mosaic, info, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 1,
                        cv2.LINE_AA)

            cv2.imshow(WINDOW_NAME, mosaic)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            # small sleep to avoid busy-wait (controls redraw rate)
            time.sleep(0.02)

    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        # allow threads to close
        time.sleep(0.5)
        cv2.destroyAllWindows()
        print("Exiting.")

if __name__ == "__main__":
    main()
