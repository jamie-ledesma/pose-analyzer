"""Exercise video analyzer: runs pose estimation and writes an annotated video.

For the live camera / upload web app, run app.py instead.

Usage:
    python main.py [input.mp4] [-o output.mp4] [--exercise squat|elbows] [--model full|heavy]
                   [--mode 3d|2d] [--max-lean DEG] [--depth DEG] [--raw] [--csv data.csv]
"""

import argparse

from pose.estimator import MODELS
from pose.pipeline import EXERCISES, Settings, analyze_file
from pose.squat import SquatThresholds


def main() -> None:
    defaults = SquatThresholds()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", nargs="?", default="input.mp4")
    parser.add_argument("-o", "--output", default="output.mp4")
    parser.add_argument("--exercise", choices=EXERCISES, default="squat",
                        help="squat: rep counter and form feedback (default); elbows: elbow angle readout")
    parser.add_argument("--model", choices=sorted(MODELS), default="full",
                        help="pose model: full (default, faster) or heavy (slower, more accurate)")
    parser.add_argument("--mode", choices=("3d", "2d"), default="3d",
                        help="elbows only: show 3d world-landmark angles (default) or 2d image-plane angles")
    parser.add_argument("--max-lean", type=float, default=defaults.max_torso_lean,
                        help=f"squat: torso lean from vertical that counts as a fault (default {defaults.max_torso_lean:g})")
    parser.add_argument("--depth", type=float, default=defaults.depth,
                        help=f"squat: knee angle a rep must reach to count (default {defaults.depth:g})")
    parser.add_argument("--raw", action="store_true", help="disable landmark smoothing (for comparison)")
    parser.add_argument("--csv", help="write per-frame measurements to this CSV file")
    args = parser.parse_args()

    settings = Settings(
        exercise=args.exercise, model=args.model, mode=args.mode, raw=args.raw,
        thresholds=SquatThresholds(max_torso_lean=args.max_lean, depth=args.depth),
    )
    report = analyze_file(args.input, args.output, settings, args.csv)

    video = report["video"]
    print(f"Input: {args.input} ({video['width']}x{video['height']} @ {video['fps']:.1f} fps, "
          f"{args.model} model, {args.exercise})")
    print(f"Output: {args.output}")
    print(f"Pose detected in {report['detected']}/{report['frames']} frames "
          f"({report['seconds']:.1f}s, {report['frames'] / report['seconds']:.1f} fps processing)")
    print(f"Landmark positions held as glitches: {report['landmarks_held']}")
    for line in report["summary"]:
        print(line)
    if args.csv:
        print(f"Data: {args.csv}")


if __name__ == "__main__":
    main()
