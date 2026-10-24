"""Generates a short synthetic test video of a figure doing squats with arm raises.

The figure is drawn with filled, skin-toned limbs and a face so BlazePose has a
chance of detecting it. It is only meant to exercise the pipeline; real footage
is needed to judge pose accuracy.

Usage:
    python tools/make_test_video.py [-o input.mp4] [--seconds 6] [--knees-cave]
"""

import argparse
import csv
import math
from pathlib import Path

import cv2
import numpy as np

W, H, FPS = 640, 720, 30

SKIN = (150, 180, 225)
SHIRT = (160, 90, 40)
PANTS = (60, 50, 45)
SHOE = (30, 30, 30)
HAIR = (40, 50, 70)


def rot(origin, length, angle_deg):
    """Point at `length` from `origin` along `angle_deg` (0 = straight down, positive = toward +x)."""
    a = math.radians(angle_deg)
    return (origin[0] + length * math.sin(a), origin[1] + length * math.cos(a))


def limb(img, p1, p2, width, color):
    p1 = tuple(int(v) for v in p1)
    p2 = tuple(int(v) for v in p2)
    cv2.line(img, p1, p2, color, width, cv2.LINE_AA)
    cv2.circle(img, p1, width // 2, color, -1, cv2.LINE_AA)
    cv2.circle(img, p2, width // 2, color, -1, cv2.LINE_AA)


def draw_figure(img, t, knees_cave=False):
    # Squat phase: 0 = standing, 1 = bottom of squat. One rep every 3 seconds.
    phase = (1 - math.cos(2 * math.pi * t / 3.0)) / 2

    cx = W / 2
    ankle_y = H - 70
    thigh, shin = 150, 150
    knee_bend = 75 * phase        # knees travel forward/outward
    # With knees_cave, the second rep (t >= 3s) collapses the knees inward instead.
    knee_dir = -0.5 if knees_cave and t >= 3.0 else 0.4
    hip_drop = 150 * phase
    hip_y = ankle_y - (thigh + shin) + hip_drop
    torso_len = 190
    torso_lean = 25 * phase        # lean forward slightly at the bottom
    arm_raise = 90 * phase        # arms raise to shoulder height

    hip_half = 38
    feet = []
    for side in (-1, 1):
        hip = (cx + side * hip_half, hip_y)
        ankle = (cx + side * 55, ankle_y)
        knee = ((hip[0] + ankle[0]) / 2 + side * knee_bend * knee_dir, (hip[1] + ankle[1]) / 2 - 10 * phase)
        limb(img, hip, knee, 46, PANTS)
        limb(img, knee, ankle, 40, PANTS)
        foot = (ankle[0] + side * 30, ankle[1] + 12)
        limb(img, ankle, foot, 22, SHOE)
        feet.append(foot)

    # Torso as a shirt-colored quad from hips to shoulders.
    neck =(cx, hip_y - torso_len * math.cos(math.radians(torso_lean)))
    shoulder_half = 62
    torso = np.array([
        (cx - hip_half - 8, hip_y + 15), (cx + hip_half + 8, hip_y + 15),
        (cx + shoulder_half, neck[1] + 10), (cx - shoulder_half, neck[1] + 10),
    ], dtype=np.int32)
    cv2.fillConvexPoly(img, torso, SHIRT, cv2.LINE_AA)

    # Arms. The elbows bend on their own 2-second cycle, independent of the squat,
    # so the angle engine sees a range of values. The left arm bends a bit further.
    curl = (1 - math.cos(2 * math.pi * t / 2.0)) / 2
    truth = {}
    for side in (-1, 1):
        shoulder = (cx + side * (shoulder_half - 8), neck[1] + 22)
        upper_angle = side * (12 + arm_raise)
        elbow = rot(shoulder, 105, upper_angle)
        bend = 10 + (100 if side == 1 else 80) * curl
        forearm_angle = upper_angle + side * bend
        wrist = rot(elbow, 95, forearm_angle)
        # The figure faces the camera, so its left arm is on the image's right (+x).
        truth["left_elbow" if side == 1 else "right_elbow"] = 180 - bend
        limb(img, shoulder, elbow, 30, SHIRT)
        limb(img, elbow, wrist, 24, SKIN)
        cv2.circle(img, tuple(int(v) for v in wrist), 16, SKIN, -1, cv2.LINE_AA)

    # Neck and head with simple facial features.
    limb(img, (cx, neck[1] + 10), (cx, neck[1] - 20), 30, SKIN)
    head = (int(cx), int(neck[1] - 65))
    cv2.ellipse(img, head, (42, 52), 0, 0, 360, SKIN, -1, cv2.LINE_AA)
    cv2.ellipse(img, (head[0], head[1] - 20), (44, 34), 0, 180, 360, HAIR, -1, cv2.LINE_AA)
    for side in (-1, 1):
        cv2.circle(img, (head[0] + side * 15, head[1] - 2), 5, (40, 30, 30), -1, cv2.LINE_AA)
    cv2.line(img, (head[0], head[1] + 2), (head[0] - 3, head[1] + 15), (110, 140, 190), 2, cv2.LINE_AA)
    cv2.ellipse(img, (head[0], head[1] + 25), (13, 5), 0, 0, 180, (80, 80, 160), 2, cv2.LINE_AA)
    return truth


def background():
    img = np.zeros((H, W, 3), np.uint8)
    for y in range(H):
        shade = int(215 - 40 * y / H)
        img[y, :] = (shade, shade, shade - 10)
    img[H - 60:, :] = (120, 135, 150)  # floor
    return img


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-o", "--output", default="input.mp4")
    parser.add_argument("--seconds", type=float, default=6.0)
    parser.add_argument("--knees-cave", action="store_true",
                        help="make the knees collapse inward on the second rep (a form fault)")
    args = parser.parse_args()

    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    truth_path = Path(args.output).with_name(Path(args.output).stem + "_truth.csv")
    bg = background()
    n = int(args.seconds * FPS)
    with open(truth_path, "w", newline="") as f:
        truth_csv = csv.writer(f)
        truth_csv.writerow(["frame", "left_elbow", "right_elbow"])
        for i in range(n):
            frame = bg.copy()
            truth = draw_figure(frame, i / FPS, args.knees_cave)
            truth_csv.writerow([i, round(truth["left_elbow"], 2), round(truth["right_elbow"], 2)])
            writer.write(frame)
    writer.release()
    print(f"Wrote {n} frames ({args.seconds:.1f}s) to {args.output}")
    print(f"Ground-truth elbow angles: {truth_path}")


if __name__ == "__main__":
    main()
