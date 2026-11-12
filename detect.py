"""Run object detection with a small pretrained YOLO model."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description="Run YOLO object detection.")
    parser.add_argument("source", help="Image, video, directory, or camera index.")
    parser.add_argument(
        "--model",
        default="yolo11n.pt",
        help="YOLO weights to use (default: yolo11n.pt).",
    )
    parser.add_argument(
        "--output",
        default="runs/detect",
        help="Directory for annotated results.",
    )
    args = parser.parse_args()

    source: str | int = int(args.source) if args.source.isdigit() else args.source
    results = YOLO(args.model).predict(
        source=source,
        save=True,
        project=Path(args.output).parent,
        name=Path(args.output).name,
    )
    print(f"Processed {len(results)} input(s). Results saved to {args.output}")


if __name__ == "__main__":
    main()
