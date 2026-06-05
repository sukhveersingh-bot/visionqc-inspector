"""FastAPI defect detection endpoint."""

from fastapi import FastAPI, File, UploadFile

from api.detector import Detector

app = FastAPI(title="QC API")
detector = Detector()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/detect")
async def detect(file: UploadFile = File(...)) -> dict:
    path = f"/tmp/{file.filename}"
    with open(path, "wb") as f:
        f.write(await file.read())
    detections = detector.predict(path)
    return {"detections": detections, "inference_ms": 0}
