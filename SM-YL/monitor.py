#!/usr/bin/env python3
"""
Virtual Line Crossing Monitor — Pure Python + OpenCV
======================================================
Model  : YOLOv8 (ultralytics) — custom weights (person=0, bag=1)
Logic  : Green (left) → safe zone. Red (right) → danger zone.
         Only Green→Red crossings are tracked.
         If a person stays in red zone > DWELL_LIMIT seconds → ALERT shown on feed.

Usage  : python monitor.py
         Press Q to quit, SPACE to pause.
"""

import cv2
import numpy as np
import time

# ═══════════════════════════════════════════════════════════════════
#  ▶ CONFIGURATION — edit these values to match your setup
# ═══════════════════════════════════════════════════════════════════

VIDEO_PATH   = "SC-yellowline.mp4"   # path to your video file
MODEL_PATH   = "best.pt"             # path to your YOLOv8 weights

CONF_THRESH  = 0.4                   # minimum detection confidence (0-1)
DWELL_LIMIT  = 10                    # seconds in red zone before alert fires
TRACK_DIST   = 90                    # max pixel distance to re-match a track
SHOW_BAGS    = True                  # also show bag detections (no crossing logic)

# Virtual yellow line — endpoints in source VIDEO pixel coordinates
# Adjust to match your scene. The LEFT side of this line = GREEN (safe),
# the RIGHT side = RED (danger).
LINE_PT1 = (640,   0)   # top point
LINE_PT2 = (640, 720)   # bottom point

# Class IDs (from your model)
CLASS_PERSON = 0
CLASS_BAG    = 1

# Colours (BGR)
CLR_GREEN  = (0,  210,  80)
CLR_RED    = (0,   40, 220)
CLR_YELLOW = (0,  230, 230)
CLR_ORANGE = (0,  150, 255)
CLR_WHITE  = (255, 255, 255)
CLR_BLACK  = (0,    0,   0)

# ═══════════════════════════════════════════════════════════════════


def side_of_line(px, py, p1, p2):
    """
    Returns +1 if point is on the LEFT  of directed line p1→p2 (GREEN zone).
             -1 if point is on the RIGHT of directed line p1→p2 (RED zone).
              0 if exactly on the line.
    Uses 2D cross-product sign.
    """
    cross = (p2[0] - p1[0]) * (py - p1[1]) - (p2[1] - p1[1]) * (px - p1[0])
    return 1 if cross > 0 else (-1 if cross < 0 else 0)


def draw_regions(frame, p1, p2, alpha=0.12):
    """Shade the two half-planes of the line (green left, red right)."""
    overlay = frame.copy()
    h, w    = frame.shape[:2]
    FAR     = max(w, h) * 5

    dx = p2[0] - p1[0];  dy = p2[1] - p1[1]
    nx = -dy;             ny =  dx          # normal pointing left (green)

    # Green half-plane
    gp = np.array([
        p1,
        p2,
        (int(p2[0] + nx * FAR), int(p2[1] + ny * FAR)),
        (int(p1[0] + nx * FAR), int(p1[1] + ny * FAR)),
    ], dtype=np.int32)
    cv2.fillPoly(overlay, [gp], (0, 140, 50))   # green fill

    # Red half-plane
    rp = np.array([
        p1,
        p2,
        (int(p2[0] - nx * FAR), int(p2[1] - ny * FAR)),
        (int(p1[0] - nx * FAR), int(p1[1] - ny * FAR)),
    ], dtype=np.int32)
    cv2.fillPoly(overlay, [rp], (0, 30, 200))   # red fill (BGR)

    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def draw_virtual_line(frame, p1, p2):
    """Draw a yellow glowing virtual line."""
    cv2.line(frame, p1, p2, (0, 180, 180), 6, cv2.LINE_AA)   # thick dark base
    cv2.line(frame, p1, p2, CLR_YELLOW, 2, cv2.LINE_AA)       # sharp yellow top

    # Endpoint circles
    for pt in (p1, p2):
        cv2.circle(frame, pt, 7, CLR_YELLOW, -1)
        cv2.circle(frame, pt, 7, CLR_BLACK,   1)

    # Region labels near midpoint
    mx = (p1[0] + p2[0]) // 2
    my = (p1[1] + p2[1]) // 2
    put_text_bg(frame, "SAFE",   (mx - 80, my), CLR_GREEN,  txt_color=CLR_BLACK,  scale=0.55)
    put_text_bg(frame, "DANGER", (mx + 10, my), CLR_RED,    txt_color=CLR_WHITE,  scale=0.55)


def draw_corner_rect(img, x1, y1, x2, y2, color, lw=2, corner=18):
    """Draw a corner-bracket style bounding box."""
    pts = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]
    dirs = [(1, 1), (-1, 1), (1, -1), (-1, -1)]
    for (px, py), (dx, dy) in zip(pts, dirs):
        cv2.line(img, (px, py), (px + dx * corner, py),          color, lw, cv2.LINE_AA)
        cv2.line(img, (px, py), (px,               py + dy * corner), color, lw, cv2.LINE_AA)
    # thin full rect
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)


def put_text_bg(img, text, org, bg, txt_color=CLR_WHITE, scale=0.45, thick=1):
    """Draw text with a filled background rectangle."""
    font  = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), bl = cv2.getTextSize(text, font, scale, thick)
    x, y = org
    cv2.rectangle(img, (x - 2, y - th - 4), (x + tw + 4, y + bl), bg, -1)
    cv2.putText(img, text, (x, y), font, scale, txt_color, thick, cv2.LINE_AA)


def draw_countdown_arc(frame, cx, cy, elapsed, limit, color):
    """Draw a circular progress arc showing dwell timer."""
    frac  = min(elapsed / limit, 1.0)
    angle = int(360 * frac)
    cv2.ellipse(frame, (cx, cy), (14, 14), -90, 0, angle, color, 3, cv2.LINE_AA)
    rem  = max(0, int(limit - elapsed))
    text = str(rem) if rem > 0 else "!"
    cv2.putText(frame, text, (cx - 6, cy + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, CLR_WHITE, 1, cv2.LINE_AA)


def draw_alert_banner(frame, msg):
    """Draw a full-width red ALERT banner at the top of the frame."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 52), (0, 0, 180), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
    # Blinking effect based on time
    if int(time.time() * 2) % 2 == 0:
        cv2.putText(frame, f"!! ALERT: {msg} !!",
                    (20, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, CLR_WHITE, 2, cv2.LINE_AA)


def draw_hud(frame, g_to_r, alert_cnt, tracked_cnt):
    """Draw a small stats HUD at the bottom-left."""
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, h - 70), (280, h), (10, 10, 10), -1)
    cv2.putText(frame, f"Green->Red crossings : {g_to_r}",
                (8, h - 48), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 230, 180), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Dwell alerts fired   : {alert_cnt}",
                (8, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 100, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Active tracks        : {tracked_cnt}",
                (8, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

def main():
    # ── Load model ────────────────────────────────────────────────
    try:
        from ultralytics import YOLO
        model = YOLO(MODEL_PATH)
        print(f"[INFO] Model loaded: {MODEL_PATH}")
    except Exception as e:
        print(f"[ERROR] Could not load model: {e}")
        return

    # ── Open video ────────────────────────────────────────────────
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {VIDEO_PATH}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    print(f"[INFO] Video: {VIDEO_PATH}  |  FPS: {fps:.1f}")
    print("[INFO] Press Q to quit, SPACE to pause/resume.")

    # ── State ─────────────────────────────────────────────────────
    tracks     = {}    # id → track dict
    next_id    = 0
    g_to_r_cnt = 0     # cumulative green→red crossing count
    alert_cnt  = 0     # cumulative dwell alerts
    paused     = False

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)   # loop
                tracks = {}                             # reset tracks on loop
                continue

        now = time.time()

        # ── Draw background regions + line ────────────────────────
        draw_regions(frame, LINE_PT1, LINE_PT2)
        draw_virtual_line(frame, LINE_PT1, LINE_PT2)

        # ── Inference ─────────────────────────────────────────────
        if not paused:
            results = model(frame, conf=CONF_THRESH, verbose=False)[0]
            persons, bags = [], []
            for box in results.boxes:
                cls  = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                if cls == CLASS_PERSON:
                    persons.append(dict(x1=x1,y1=y1,x2=x2,y2=y2,cx=cx,cy=cy,conf=conf,matched=False))
                elif cls == CLASS_BAG and SHOW_BAGS:
                    bags.append(dict(x1=x1,y1=y1,x2=x2,y2=y2,conf=conf))

            # ── Nearest-neighbour tracker ──────────────────────────
            new_tracks = {}

            for id_, tr in tracks.items():
                best, best_d = None, TRACK_DIST
                for p in persons:
                    if p['matched']:
                        continue
                    d = ((p['cx'] - tr['cx'])**2 + (p['cy'] - tr['cy'])**2) ** 0.5
                    if d < best_d:
                        best_d, best = d, p

                if best is None:
                    continue   # track lost — drop it

                best['matched'] = True
                new_side          = side_of_line(best['cx'], best['cy'], LINE_PT1, LINE_PT2)
                old_side          = tr['side']
                crossed_to_red_at = tr['crossed_to_red_at']
                alerted           = tr['alerted']

                # ── Crossing: GREEN → RED ──────────────────────────
                if old_side == 1 and new_side == -1:
                    crossed_to_red_at = now
                    alerted           = False
                    g_to_r_cnt       += 1
                    print(f"[CROSS] Person #{id_:03d}  GREEN → RED  "
                          f"({best['cx']},{best['cy']})")

                # ── Returned to GREEN → reset timer ────────────────
                if new_side == 1 and crossed_to_red_at is not None:
                    crossed_to_red_at = None
                    alerted           = False

                # ── Dwell alert ────────────────────────────────────
                if (new_side == -1 and crossed_to_red_at is not None
                        and not alerted
                        and (now - crossed_to_red_at) >= DWELL_LIMIT):
                    alerted    = True
                    alert_cnt += 1
                    print(f"[ALERT] Person #{id_:03d} in RED zone for "
                          f"{DWELL_LIMIT}s! (total alerts: {alert_cnt})")

                new_tracks[id_] = dict(
                    cx=best['cx'], cy=best['cy'],
                    x1=best['x1'], y1=best['y1'], x2=best['x2'], y2=best['y2'],
                    conf=best['conf'],
                    side=new_side if new_side != 0 else old_side,
                    crossed_to_red_at=crossed_to_red_at,
                    alerted=alerted,
                )

            # New unmatched detections → new tracks
            for p in persons:
                if p['matched']:
                    continue
                s = side_of_line(p['cx'], p['cy'], LINE_PT1, LINE_PT2)
                new_tracks[next_id] = dict(
                    cx=p['cx'], cy=p['cy'],
                    x1=p['x1'], y1=p['y1'], x2=p['x2'], y2=p['y2'],
                    conf=p['conf'],
                    side=s,
                    crossed_to_red_at=None,
                    alerted=False,
                )
                next_id += 1

            tracks = new_tracks

        # ── Draw tracked persons ───────────────────────────────────
        any_alert = False

        for id_, tr in tracks.items():
            x1, y1, x2, y2 = tr['x1'], tr['y1'], tr['x2'], tr['y2']
            cx, cy = tr['cx'], tr['cy']

            in_red    = (tr['side'] == -1)
            has_dwell = (tr['crossed_to_red_at'] is not None)
            alerted   = tr['alerted']

            # Colour: red as soon as they cross G→R, green otherwise
            color = CLR_RED if (in_red and has_dwell) else CLR_GREEN

            if alerted and in_red:
                any_alert = True

            # Bounding box
            draw_corner_rect(frame, x1, y1, x2, y2, color, lw=2)

            # Countdown arc (red box, counting down, not yet alerted)
            if in_red and has_dwell and not alerted:
                elapsed = now - tr['crossed_to_red_at']
                draw_countdown_arc(frame, cx, cy, elapsed, DWELL_LIMIT, CLR_RED)

            # ⚠ text for alerted persons
            if alerted and in_red:
                cv2.putText(frame, "⚠ ALERT", (x1, y1 - 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, CLR_RED, 2, cv2.LINE_AA)

            # Label: ID + confidence + dwell time
            dwell_str = ""
            if in_red and has_dwell:
                elapsed   = now - tr['crossed_to_red_at']
                dwell_str = f"  T+{elapsed:.0f}s{'  !ALERT' if alerted else ''}"
            label = f"P#{id_:03d} {tr['conf']:.0%}{dwell_str}"
            put_text_bg(frame, label, (x1, y1 - 4), color,
                        txt_color=CLR_WHITE if (in_red and has_dwell) else CLR_BLACK)

        # ── Draw bags (detection only, no crossing logic) ──────────
        for b in bags:
            cv2.rectangle(frame, (b['x1'],b['y1']), (b['x2'],b['y2']), CLR_ORANGE, 1)
            put_text_bg(frame, f"Bag {b['conf']:.0%}",
                        (b['x1'], b['y1'] - 4), CLR_ORANGE,
                        txt_color=CLR_WHITE, scale=0.4)

        # ── Alert banner (top of frame) ────────────────────────────
        if any_alert:
            n_alerted = sum(1 for t in tracks.values() if t['alerted'] and t['side'] == -1)
            draw_alert_banner(frame,
                              f"{n_alerted} person(s) lingering in RED zone > {DWELL_LIMIT}s")

        # ── HUD (bottom-left) ─────────────────────────────────────
        draw_hud(frame, g_to_r_cnt, alert_cnt, len(tracks))

        # ── Paused indicator ──────────────────────────────────────
        if paused:
            h, w = frame.shape[:2]
            put_text_bg(frame, "  PAUSED — press SPACE to resume  ",
                        (w // 2 - 170, 90), (40, 40, 40), scale=0.6)

        # ── Show frame ────────────────────────────────────────────
        cv2.imshow("Virtual Line Monitor", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n[DONE]  G→R crossings: {g_to_r_cnt}  |  Dwell alerts: {alert_cnt}")


if __name__ == "__main__":
    main()
