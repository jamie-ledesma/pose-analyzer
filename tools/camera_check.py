"""Checks that the camera works: how long the first frame takes and whether frames keep coming.

Run from the terminal app you start the server from (macOS gives camera access per app):
    .venv/bin/python tools/camera_check.py [--720p] [camera index]
"""

import sys
import time

import cv2

args = [a for a in sys.argv[1:] if not a.startswith("--")]
index = int(args[0]) if args else 0
cap = cv2.VideoCapture(index)
if "--720p" in sys.argv:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
print("opened:", cap.isOpened())

start, first, frames, last_ok, longest_gap = time.time(), None, 0, None, 0.0
while cap.isOpened() and time.time() - start < 8:
    ok, frame = cap.read()
    now = time.time()
    if ok:
        if first is None:
            first = now - start
            print(f"first frame after {first:.2f} s, size {frame.shape[1]}x{frame.shape[0]}")
        elif last_ok is not None:
            longest_gap = max(longest_gap, now - last_ok)
        frames, last_ok = frames + 1, now
    else:
        time.sleep(0.02)
cap.release()
print(f"{frames} frames in 8 s, longest gap between frames {longest_gap:.2f} s")
