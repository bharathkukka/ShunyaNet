"""
pyqt_rtsp_mosaic.py

PyQt5 application that ingests multiple RTSP streams and displays them in a clickable
mosaic grid. Click a tile to open it full-screen (or double-click to toggle). Press Esc
or click the close button to return to grid view.

Features:
- Threaded RTSP readers using OpenCV (FFMPEG backend by default)
- Single-slot per-stream latest-frame storage (drop-old strategy)
- PyQt5 GUI with grid layout of QLabel tiles showing live frames
- Click a tile to view full-screen; press Esc to go back
- Configurable RTSP_URLS, tile size, and polling FPS at the top of the file

Dependencies:
- Python 3.8+
- PyQt5
- opencv-python
- numpy

Run:
    python3 pyqt_rtsp_mosaic.py

Edit RTSP_URLS in the USER CONFIG section to point to your streams (or local MediaMTX test streams).
"""

import sys
import threading
import queue
import time
import math
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QGridLayout, QVBoxLayout, QHBoxLayout,
    QPushButton, QSizePolicy, QMainWindow, QScrollArea
)
from PyQt5.QtGui import QImage, QPixmap, QMouseEvent
from PyQt5.QtCore import QTimer, Qt

# --------------------- USER CONFIG ---------------------
# Replace with your RTSP URLs or generate e.g. rtsp://localhost:8554/stream1...
RTSP_URLS = [f"rtsp://localhost:8554/stream{i}" for i in range(1, 11)]  # example: 12 streams
TILE_SIZE = (320, 180)   # width, height for each tile
POLL_FPS = 6             # per-stream read throttle (lower -> lighter CPU)
QUEUE_MAX = 1            # keep only latest frame
RECONNECT_INTERVAL = 2.0 # seconds between reconnect attempts
WINDOW_TITLE = "RTSP Mosaic - PyQt"
# -------------------------------------------------------


class StreamReader(threading.Thread):
    """Thread that reads frames from an RTSP URL and keeps only the latest frame in a queue."""
    def __init__(self, url, frame_slot, stop_event, poll_fps=POLL_FPS):
        super().__init__(daemon=True)
        self.url = url
        self.frame_slot = frame_slot
        self.stop_event = stop_event
        self.poll_interval = 1.0 / poll_fps if poll_fps > 0 else 0

    def open_capture(self):
        # Try to open with FFMPEG backend; customize here if you want GStreamer pipelines
        cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        return cap

    def run(self):
        cap = self.open_capture()
        while not self.stop_event.is_set():
            if not cap or not cap.isOpened():
                # try reconnect
                try:
                    cap.release()
                except Exception:
                    pass
                time.sleep(RECONNECT_INTERVAL)
                cap = self.open_capture()
                continue

            ret, frame = cap.read()
            if not ret or frame is None:
                # short sleep then continue to try reading
                time.sleep(0.1)
                continue

            # put newest frame into single-slot queue (drop previous)
            try:
                try:
                    _ = self.frame_slot.get_nowait()
                except queue.Empty:
                    pass
                self.frame_slot.put_nowait(frame)
            except queue.Full:
                pass

            if self.poll_interval > 0:
                time.sleep(self.poll_interval)

        try:
            cap.release()
        except Exception:
            pass


class ClickableLabel(QLabel):
    """QLabel that emits click events with an index integer stored in property 'idx'."""
    def __init__(self, idx, parent=None):
        super().__init__(parent)
        self.idx = idx
        self.setScaledContents(True)

    def mousePressEvent(self, ev: QMouseEvent):
        # Left click - show fullscreen; Right click - placeholder for context menu
        if ev.button() == Qt.LeftButton:
            # emit custom event by calling parent handler if present
            if hasattr(self.parent(), 'on_tile_clicked'):
                self.parent().on_tile_clicked(self.idx)
        super().mousePressEvent(ev)


class FullScreenWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout = QVBoxLayout(self)
        layout.addWidget(self.label)
        self.setLayout(layout)
        self.setWindowFlags(Qt.Window)
        self.showFullScreen()

    def set_frame(self, frame):
        if frame is None:
            self.label.setText("No frame")
            return
        h, w = frame.shape[:2]
        bytes_per_line = 3 * w
        image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_BGR888)
        pix = QPixmap.fromImage(image)
        self.label.setPixmap(pix.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()
        super().keyPressEvent(event)


class MosaicWindow(QMainWindow):
    def __init__(self, rtsp_urls, tile_size=TILE_SIZE):
        super().__init__()
        self.setWindowTitle(WINDOW_TITLE)
        self.rtsp_urls = rtsp_urls
        self.n = len(rtsp_urls)
        self.tile_w, self.tile_h = tile_size

        # compute grid
        self.rows, self.cols = self.compute_grid(self.n)

        # prepare frame storage: a single-slot queue per stream
        self.frame_slots = [queue.Queue(maxsize=QUEUE_MAX) for _ in range(self.n)]
        self.stop_event = threading.Event()
        self.threads = []

        # UI: central widget with grid
        central = QWidget()
        self.setCentralWidget(central)

        self.grid_layout = QGridLayout()
        self.grid_layout.setSpacing(4)

        # Scroll area so large grids can be scrolled
        scroll = QScrollArea()
        content = QWidget()
        content.setLayout(self.grid_layout)
        scroll.setWidget(content)
        scroll.setWidgetResizable(True)

        vbox = QVBoxLayout()
        vbox.addWidget(scroll)

        # control bar
        ctrl = QHBoxLayout()
        self.btn_quit = QPushButton("Quit")
        self.btn_quit.clicked.connect(self.close)
        ctrl.addWidget(self.btn_quit)
        ctrl.addStretch(1)
        vbox.addLayout(ctrl)

        central.setLayout(vbox)

        # create tiles
        self.tiles = []
        for i in range(self.n):
            lbl = ClickableLabel(i, parent=self)
            lbl.setFixedSize(self.tile_w, self.tile_h)
            lbl.setStyleSheet("background-color: #111; color: #ddd; border: 1px solid #333;")
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setText(f"Stream {i+1}\n{self.rtsp_urls[i]}")
            self.tiles.append(lbl)
            r = i // self.cols
            c = i % self.cols
            self.grid_layout.addWidget(lbl, r, c)

        # start readers
        for i, url in enumerate(self.rtsp_urls):
            t = StreamReader(url, self.frame_slots[i], self.stop_event, poll_fps=POLL_FPS)
            t.start()
            self.threads.append(t)

        # fullscreen window (created on demand)
        self.fullwin = None
        self.full_index = None

        # QTimer to refresh UI from latest frames
        self.timer = QTimer()
        self.timer.timeout.connect(self.refresh_tiles)
        self.timer.start(int(1000 / max(10, POLL_FPS)))  # GUI refresh faster than read fps

    def compute_grid(self, n):
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)
        return rows, cols

    def on_tile_clicked(self, idx):
        # open full-screen view for tile idx
        self.full_index = idx
        if self.fullwin is None or not self.fullwin.isVisible():
            self.fullwin = FullScreenWindow(self)
            self.fullwin.show()

    def refresh_tiles(self):
        # update each tile's pixmap from latest frame if available
        for i, q in enumerate(self.frame_slots):
            frame = None
            try:
                # non-blocking get latest
                frame = q.get_nowait()
            except queue.Empty:
                frame = None

            if frame is not None:
                # prepare QPixmap
                thumb = cv2.resize(frame, (self.tile_w, self.tile_h), interpolation=cv2.INTER_AREA)
                h, w = thumb.shape[:2]
                bytes_per_line = 3 * w
                qimg = QImage(thumb.data, w, h, bytes_per_line, QImage.Format_BGR888)
                pix = QPixmap.fromImage(qimg)
                self.tiles[i].setPixmap(pix)

                # if full-screen open for this index, update it too
                if self.fullwin and self.full_index == i and self.fullwin.isVisible():
                    self.fullwin.set_frame(frame)

    def closeEvent(self, event):
        # cleanup threads
        self.stop_event.set()
        # allow small grace period
        time.sleep(0.2)
        event.accept()


def main():
    app = QApplication(sys.argv)
    win = MosaicWindow(RTSP_URLS)
    win.resize(1200, 800)
    win.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
