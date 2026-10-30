"""Label same physical points before running DINOv3, without loading a model."""

import argparse
import json
from pathlib import Path

from mot17_dinov3_point_match import (choose_point, choose_roi, crop_roi,
                                      image_at_frame, load_pairs)


def parse_pair(value):
    try:
        a, b = (int(n) for n in value.split(":"))
        if a < 1 or b < 1 or a == b:
            raise ValueError
        return a, b
    except ValueError as error:
        raise argparse.ArgumentTypeError("Use SOURCE:TARGET, e.g. 61:76") from error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--sequence-dir", type=Path)
    source.add_argument("--video", type=Path)
    parser.add_argument("--frames", type=parse_pair, nargs="+", required=True,
                        help="1-based MOT17 frame pairs, e.g. 61:76 91:106")
    parser.add_argument("--points-per-pair", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("pairs.json"))
    parser.add_argument("--append", action="store_true", help="explicitly add new labels to an existing file")
    args = parser.parse_args()
    if args.points_per_pair < 1:
        parser.error("--points-per-pair must be positive")
    if args.sequence_dir and not (args.sequence_dir / "img1").is_dir():
        parser.error("--sequence-dir must contain img1/")
    if args.video and not args.video.is_file():
        parser.error("--video does not exist")
    import cv2

    if args.output.exists() and not args.append:
        parser.error("annotations already exist; choose a new --output or pass --append")
    existing = load_pairs(args.output) if args.output.exists() else []

    def frame_at(number):
        if args.sequence_dir:
            file = args.sequence_dir / "img1" / f"{number:06d}.jpg"
            frame = cv2.imread(str(file))
            if frame is None:
                raise FileNotFoundError(file)
            return frame
        return image_at_frame(args.video, number)

    for a, b in args.frames:
        first, second = frame_at(a), frame_at(b)
        print(f"Frames {a} and {b}: select the SAME person's ROI in both frames")
        roi0 = choose_roi(first, f"Frame {a} person")
        roi1 = choose_roi(second, f"Frame {b} same person")
        crop0, crop1 = crop_roi(first, roi0), crop_roi(second, roi1)
        for j in range(args.points_per_pair):
            print(f"Point {j + 1}/{args.points_per_pair}: choose a visible physical detail (pattern, bag logo, etc.)")
            p0 = choose_point(crop0, f"frame {a} source")
            print("Now click precisely the SAME detail in target frame. Skip pairs with occluded/invisible details.")
            p1 = choose_point(crop1, f"frame {b} GT")
            existing.append({"source_frame": a, "target_frame": b,
                             "source_roi": list(roi0), "target_roi": list(roi1),
                             "source_point": list(p0), "target_point": list(p1)})
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"Saved {len(existing)} labelled points to {args.output}")


if __name__ == "__main__":
    main()
