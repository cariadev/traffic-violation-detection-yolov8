# traffic_api.py — v2
# Thay đổi so với v1 (mock):
#   - Load model biển báo thật (YOLOv8) để detect signs từ ảnh thực
#   - Đèn giao thông vẫn dùng chu kỳ thời gian thực (không đổi)
#   - Signs trả về bbox + class + conf thật từ model

import time
import base64
import numpy as np
import cv2
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from ultralytics import YOLO

# ── Đường dẫn model biển báo ─────────────────────────────────────────── #
SIGN_MODEL_PATH = r"E:\HK2-2025-2026\Học Sâu\Semester\runs\traffic-sign\yolov8n_adamw\weights\best.pt"
SIGN_CONF       = 0.40   # Ngưỡng confidence tối thiểu

# ── Load model một lần khi khởi động server ──────────────────────────── #
print(f"[SIGN MODEL] Đang tải: {SIGN_MODEL_PATH}")
try:
    sign_model = YOLO(SIGN_MODEL_PATH)
    SIGN_NAMES = sign_model.names   # dict {id: "ten_bien"}
    print(f"[SIGN MODEL] Tải thành công. Classes: {SIGN_NAMES}")
except Exception as e:
    sign_model = None
    SIGN_NAMES = {}
    print(f"[SIGN MODEL] CẢNH BÁO: Không tải được model: {e}")

app = FastAPI(
    title="Traffic Detection API",
    description="Phát hiện đèn giao thông (chu kỳ thời gian) + biển báo (YOLOv8 thật)"
)


class PredictRequest(BaseModel):
    image_b64: str
    detect: list[str] = ["light", "sign"]


def _decode_image(image_b64: str) -> np.ndarray:
    """Giải mã base64 → numpy BGR image."""
    buf   = base64.b64decode(image_b64)
    arr   = np.frombuffer(buf, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return frame


def _get_light_color() -> str:
    """
    Chu kỳ đèn dựa theo thời gian thực:
      0–9s   → GREEN
      10–12s → YELLOW
      13–24s → RED
    """
    cycle = int(time.time()) % 25
    if cycle < 10:
        return "GREEN"
    elif cycle < 13:
        return "YELLOW"
    else:
        return "RED"


def _detect_signs(frame: np.ndarray) -> list[dict]:
    """Chạy sign model, trả về list {bbox, class, conf}."""
    if sign_model is None or frame is None:
        return []

    results = sign_model.predict(frame, conf=SIGN_CONF, verbose=False)
    signs   = []

    if results and results[0].boxes is not None:
        for box in results[0].boxes:
            cls_id   = int(box.cls[0])
            cls_name = SIGN_NAMES.get(cls_id, f"sign_{cls_id}")
            conf     = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            signs.append({
                "bbox":  [x1, y1, x2, y2],
                "class": cls_name,
                "conf":  round(conf, 3),
            })

    return signs


@app.post("/predict")
def predict(request: PredictRequest):
    result = {}

    # ── Đèn giao thông ───────────────────────────────────────────────── #
    if "light" in request.detect:
        color = _get_light_color()
        result["light"] = {
            "color":  color,
            "source": "Time-based cycle",
        }

    # ── Biển báo (model thật) ─────────────────────────────────────────── #
    if "sign" in request.detect:
        try:
            frame = _decode_image(request.image_b64)
            signs = _detect_signs(frame)
        except Exception as e:
            print(f"[SIGN DETECT] Lỗi: {e}")
            signs = []

        result["signs"] = signs

        if signs:
            print(f"[SIGN DETECT] Phát hiện {len(signs)} biển: "
                  f"{[s['class'] for s in signs]}")

    return result


if __name__ == "__main__":
    print("------------------------------------------------------------")
    print("🚦 Traffic Detection API — Port 8502")
    print("💡 Đèn: 10s Xanh → 3s Vàng → 12s Đỏ (chu kỳ 25s)")
    print(f"🪧 Sign model: {SIGN_MODEL_PATH}")
    print("------------------------------------------------------------")
    uvicorn.run(app, host="0.0.0.0", port=8502)