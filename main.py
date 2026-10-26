"""
YOLOv8 Object Detection System
A beginner-friendly project using the Ultralytics YOLOv8 model.
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def parse_source(source_text: str):
    """Convert webcam source '0' to integer 0; leave file paths as strings."""
    if source_text.isdigit():
        return int(source_text)
    return source_text


def main():
    parser = argparse.ArgumentParser(
        description="Detect objects in an image, video, or webcam stream using YOLOv8."
    )
    parser.add_argument(
        "--source",
        default="input/test.jpg",
        help="Image/video path or webcam number. Default: input/test.jpg",
    )
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="YOLO model weights. Default: yolov8n.pt",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="Minimum confidence threshold from 0.0 to 1.0. Default: 0.25",
    )
    parser.add_argument(
        "--output",
        default="output",
        help="Folder where detection results are saved. Default: output",
    )
    args = parser.parse_args()

    if not 0.0 <= args.confidence <= 1.0:
        parser.error("--confidence must be between 0.0 and 1.0")

    source = parse_source(args.source)
    if isinstance(source, str) and not Path(source).exists():
        parser.error(
            f"Source file not found: {source}\n"
            "Place an image at input/test.jpg or run with --source path/to/image.jpg"
        )

    print(f"Loading model: {args.model}")
    model = YOLO(args.model)  # Downloads the pretrained weights on first use if needed.

    print(f"Running detection on: {source}")
    results = model.predict(
        source=source,
        conf=args.confidence,
        save=True,
        project=args.output,
        name="results",
        exist_ok=True,
        show=False,
    )

    total = 0
    for result in results:
        count = len(result.boxes) if result.boxes is not None else 0
        total += count
        print(f"Detected {count} object(s) in {result.path}")

        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                label = model.names[class_id]
                print(f"  - {label}: {confidence:.2%} confidence")

    print(f"\nDetection finished. Results are saved in: {Path(args.output) / 'results'}")
    print(f"Total detections across processed frame(s): {total}")


if __name__ == "__main__":
    main()
