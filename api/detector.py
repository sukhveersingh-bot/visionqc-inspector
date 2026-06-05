"""YOLO detector wrapper."""

from pathlib import Path

from ultralytics import YOLO

DEFAULT = Path(__file__).resolve().parents[1] / "models" / "best.onnx"


class Detector:
    def __init__(self, weights: Path = DEFAULT) -> None:
        self.model = YOLO(str(weights)) if weights.exists() else None

    def predict(self, image_path: str) -> list[dict]:
        if self.model is None:
            return []
        results = self.model.predict(image_path, verbose=False)
        out = []
        for r in results:
            for box in r.boxes:
                out.append(
                    {
                        "class": r.names[int(box.cls)],
                        "confidence": float(box.conf),
                        "bbox": box.xyxy[0].tolist(),
                    }
                )
        return out
