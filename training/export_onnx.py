"""Export YOLOv8 weights to ONNX for CPU inference."""

from pathlib import Path

from ultralytics import YOLO

WEIGHTS = Path("models/best.pt")


def main() -> None:
    model = YOLO(str(WEIGHTS))
    model.export(format="onnx", dynamic=True)


if __name__ == "__main__":
    main()
