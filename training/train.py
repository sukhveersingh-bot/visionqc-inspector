"""Train YOLOv8 on defect dataset."""

from ultralytics import YOLO


def main() -> None:
    model = YOLO("yolov8n.pt")
    model.train(data="data/roboflow/data.yaml", epochs=50, imgsz=640, project="runs")


if __name__ == "__main__":
    main()
