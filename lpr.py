# lpr.py — Nhận diện biển số xe bằng EasyOCR (v4)
# v4: multi-scale crop thông minh hơn, thử nhiều vùng biển số,
#     lấy frame vàng (xe nửa qua vạch) → biển rõ hơn

import cv2
import re
import numpy as np

_reader = None


def _get_reader():
    global _reader
    if _reader is None:
        import easyocr
        _reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    return _reader


def preprocess_plate(img):
    """
    Tăng cường ảnh cho camera xa:
    - CLAHE clipLimit 3.0
    - Upscale thông minh đến tối thiểu h=80px
    - Sharpen để OCR đọc nét cạnh biển tốt hơn
    - v4: thêm thử binary threshold song song
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # CLAHE
    clahe    = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(gray)

    # Upscale nếu ảnh quá nhỏ
    h, w = enhanced.shape[:2]
    if h < 80 or w < 120:
        scale    = max(2, int(np.ceil(80 / max(h, 1))))
        enhanced = cv2.resize(enhanced, (w * scale, h * scale),
                              interpolation=cv2.INTER_CUBIC)

    # Sharpen
    kernel   = np.array([[0, -1, 0],
                         [-1, 5, -1],
                         [0, -1, 0]], dtype=np.float32)
    enhanced = cv2.filter2D(enhanced, -1, kernel)
    enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
    return enhanced


def preprocess_plate_binary(img):
    """
    Bản thay thế: binary threshold — đôi khi hiệu quả hơn với biển tối/sáng tương phản cao.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    if h < 80 or w < 120:
        scale = max(2, int(np.ceil(80 / max(h, 1))))
        gray  = cv2.resize(gray, (w * scale, h * scale),
                           interpolation=cv2.INTER_CUBIC)
    # Adaptive threshold
    binary = cv2.adaptiveThreshold(gray, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    return binary


def clean_plate_text(raw: str) -> str:
    """
    Chuẩn hóa biển số VN: 2 số + 1-2 chữ + gạch + 3-5 số
    Ví dụ: 51F-123.45, 43A1-123.45
    """
    text = raw.upper()
    text = text.replace(" ", "").replace("_", "")
    # Sửa ký tự OCR hay nhầm
    text = text.replace("O", "0").replace("I", "1").replace("l", "1")
    text = re.sub(r"[^A-Z0-9\-\.]", "", text)

    # Ưu tiên match pattern biển số VN chuẩn
    vn_pattern = re.search(r"(\d{2}[A-Z]{1,2}[-]?\d{3,5})", text)
    if vn_pattern:
        result = vn_pattern.group(1)
        # Thêm gạch ngang nếu thiếu
        result = re.sub(r"(\d{2}[A-Z]{1,2})(\d{3,5})", r"\1-\2", result)
        return result

    if 5 <= len(text) <= 12 and any(c.isdigit() for c in text):
        return text
    return ""


def read_plate(frame: np.ndarray, vehicle_bbox: tuple) -> str:
    """
    v4: Multi-scale crop thông minh.

    Các vùng crop theo tỷ lệ chiều cao bbox xe:
      - 60-100%: biển phổ thông (xe máy, ô tô con) — vùng chính
      - 50-80%:  xe lớn có cabin cao (tải, buýt)
      - 70-100%: xe máy biển gắn thấp
      - 55-85%:  thêm vùng trung gian

    Với mỗi vùng crop, thử 2 kiểu preprocess (CLAHE + binary).
    Lấy kết quả có confidence cao nhất.
    """
    x1, y1, x2, y2 = [int(v) for v in vehicle_bbox]
    h  = y2 - y1
    fh, fw = frame.shape[:2]

    crop_regions = [
        (y1 + int(h * 0.60), y2),           # biển phổ thông
        (y1 + int(h * 0.50), y1 + int(h * 0.80)),  # xe lớn
        (y1 + int(h * 0.70), y2),           # xe máy biển thấp
        (y1 + int(h * 0.55), y1 + int(h * 0.85)),  # trung gian
    ]

    reader    = _get_reader()
    best_text = ""
    best_conf = 0.0

    for py1, py2 in crop_regions:
        py1 = max(0, py1)
        py2 = min(fh, py2)
        cx1 = max(0, x1)
        cx2 = min(fw, x2)

        if py2 <= py1 or cx2 <= cx1:
            continue

        plate_img = frame[py1:py2, cx1:cx2]
        if plate_img.shape[0] < 10 or plate_img.shape[1] < 20:
            continue

        # Thử 2 kiểu preprocess
        for proc_fn in [preprocess_plate, preprocess_plate_binary]:
            try:
                processed = proc_fn(plate_img)
            except Exception:
                continue

            try:
                results = reader.readtext(
                    processed,
                    detail=1,
                    allowlist="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-."
                )
                for (_, text, conf) in results:
                    if conf < 0.35:   # v4: 0.4→0.35 để không bỏ sót
                        continue
                    cleaned = clean_plate_text(text)
                    if cleaned and conf > best_conf:
                        best_conf = conf
                        best_text = cleaned
                        print(f"[LPR] '{text}' → '{cleaned}' "
                              f"conf={conf:.2f} fn={proc_fn.__name__}")
            except Exception as e:
                print(f"[LPR] OCR error: {e}")
                continue

    if best_text:
        print(f"[LPR] Kết quả cuối: '{best_text}' conf={best_conf:.2f}")
    else:
        print(f"[LPR] Không đọc được biển → trả về rỗng")

    return best_text