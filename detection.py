# detection.py — v17
# Thay đổi so với v16:
#   - SỬA BUG CHÍNH: không bắt được xe vượt đèn đỏ
#       * Bỏ _is_moving_down() trong logic vượt đèn đỏ
#         → xe đứng chờ rồi nhích qua vạch: history y gần bằng nhau → _is_moving_down luôn False
#       * Thay bằng: chỉ cần (v_y2 - before_y_red) > MIN_CROSSING_MOVE_PX là vi phạm
#         → tức là xe đã dịch chuyển TỔNG CỘNG đủ xa so với vị trí trước vạch lúc đèn đỏ
#       * _confirmed_before đổi từ set → dict[tid, float] lưu y2 lớn nhất trước vạch
#       * Giảm MIN_CROSSING_MOVE_PX = 15px (đủ tránh bbox jitter, đủ nhạy với xe chậm)
#       * Giảm MIN_TRACK_BEFORE_VIOLATION = 3
#   - _is_moving_down đã xóa hoàn toàn (không cần thiết)
#   - Thêm [SKIP] log để debug khi xe bị bỏ qua
#   - STOP_LINE_Y_RATIO = 0.60 (giữ nguyên)

import cv2
import numpy as np
import os
import base64
import requests
import threading
import time
from collections import defaultdict
from datetime import datetime
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont

from rules import (RULE_MAP, VEHICLE_CLASS_NAMES,
                   PROXIMITY_THRESHOLD, MIN_TRACK_FRAMES, MIN_MOVE_PX)
from lpr import read_plate
from database import insert_violation, IMG_DIR

# ── Paths ────────────────────────────────────────────────────────────── #
VEHICLE_MODEL_PATH = r"E:\HK2-2025-2026\Học Sâu\traffic_system\model\best.pt"

# ── API ──────────────────────────────────────────────────────────────── #
TRAFFIC_API_URL    = "http://localhost:8502/predict"
LIGHT_API_INTERVAL = 4
API_RESIZE_W       = 640

# ── Logic constants ───────────────────────────────────────────────────── #
RED_CONFIRM_FRAMES         = 2
STOP_LINE_Y_RATIO          = 0.60
# Xe phải vượt qua vạch ít nhất N pixel (tính từ vị trí lúc đèn đỏ) mới bắt
# 15px = đủ tránh jitter bbox, đủ nhạy với xe nhích chậm
MIN_CROSSING_MOVE_PX       = 15
MIN_TRACK_BEFORE_VIOLATION = 3
FRAME_SKIP                 = 2
CROP_ZOOM_FACTOR           = 2.5
CROP_PADDING               = 30

os.makedirs(IMG_DIR, exist_ok=True)


# ── Font ─────────────────────────────────────────────────────────────── #
def _load_font(size: int):
    for path in [r"C:\Windows\Fonts\arial.ttf",
                 r"C:\Windows\Fonts\segoeui.ttf",
                 r"C:\Windows\Fonts\tahoma.ttf"]:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()

FONT_SM = _load_font(16)
FONT_MD = _load_font(20)
FONT_LG = _load_font(24)


# ── TextOverlay ──────────────────────────────────────────────────────── #
class TextOverlay:
    def __init__(self, img: np.ndarray):
        self._img  = img
        self._cmds = []

    def add(self, text, pos, font=None, color=(255, 255, 255)):
        if font is None:
            font = FONT_MD
        self._cmds.append((text, pos, font, (color[2], color[1], color[0])))
        return self

    def render(self) -> np.ndarray:
        if not self._cmds:
            return self._img
        pil  = Image.fromarray(cv2.cvtColor(self._img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil)
        for text, pos, font, color_rgb in self._cmds:
            draw.text(pos, text, font=font, fill=color_rgb)
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def put_text_vn(img, text, pos, font=None, color=(255, 255, 255)):
    return TextOverlay(img).add(text, pos, font, color).render()


# ── Plate helpers ─────────────────────────────────────────────────────── #
def _is_valid_plate(plate: str) -> bool:
    if not plate:
        return False
    cleaned = "".join(c for c in plate if c.isascii() and (c.isalnum() or c in "-."))
    return 5 <= len(cleaned) <= 10

def _clean_plate(plate: str) -> str:
    return "".join(c for c in plate if c.isascii() and (c.isalnum() or c in "-."))


# ── API helpers ───────────────────────────────────────────────────────── #
def _resize_for_api(frame: np.ndarray, max_w: int = API_RESIZE_W) -> np.ndarray:
    h, w = frame.shape[:2]
    if w <= max_w:
        return frame
    scale = max_w / w
    return cv2.resize(frame, (max_w, int(h * scale)), interpolation=cv2.INTER_AREA)

def _call_traffic_api(frame: np.ndarray, detect=None) -> dict:
    if detect is None:
        detect = ["light", "sign"]
    small  = _resize_for_api(frame)
    _, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 75])
    b64    = base64.b64encode(buf).decode("utf-8")
    try:
        resp = requests.post(
            TRAFFIC_API_URL,
            json={"image_b64": b64, "detect": detect},
            timeout=2.0,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        return {}
    except Exception as e:
        print(f"[TRAFFIC API] Loi: {e}")
        return {}

def _parse_api_response(api_data: dict) -> tuple[str, str, list]:
    if not api_data:
        return "OFF", "none", []

    light_info = api_data.get("light", api_data)
    raw_color  = str(light_info.get("color", "OFF")).upper().strip()
    source     = light_info.get("source", "?")
    signs      = api_data.get("signs", [])

    color_map = {
        "RED": "RED", "GREEN": "GREEN", "YELLOW": "YELLOW",
        "DO":  "RED", "XANH": "GREEN", "VANG":   "YELLOW",
        "OFF": "OFF", "UNKNOWN": "OFF", "NONE": "OFF",
    }
    mapped = color_map.get(raw_color, "OFF")
    if raw_color not in color_map:
        print(f"[PARSE] Mau khong nhan ra: '{raw_color}' -> OFF")
    return mapped, source, signs


# ════════════════════════════════════════════════════════════════════════ #
class ViolationDetector:
    def __init__(self,
                 vehicle_model_path=VEHICLE_MODEL_PATH,
                 location="Camera 1",
                 conf=0.35):

        # ── Load vehicle model ──────────────────────────────────────── #
        try:
            self.vehicle_model = YOLO(vehicle_model_path)
            print(f"[DETECTOR v17] Da tai model xe: {vehicle_model_path}")
        except Exception as e:
            print(f"[DETECTOR v17] Khong tai duoc model xe: {e} -> dung yolov8n.pt")
            self.vehicle_model = YOLO("yolov8n.pt")

        # ── Build VEHICLE_CLASSES tu model.names dong ──────────────── #
        self.vehicle_classes: set[int] = set()
        for cls_id, cls_name in self.vehicle_model.names.items():
            if cls_name in VEHICLE_CLASS_NAMES:
                self.vehicle_classes.add(cls_id)
        print(f"[DETECTOR v17] Vehicle classes: "
              f"{ {i: self.vehicle_model.names[i] for i in self.vehicle_classes} }")

        self.location = location
        self.conf     = conf

        # ── State dicts ─────────────────────────────────────────────── #
        self.track_history:   dict[int, list]       = defaultdict(list)
        self.logged:          dict[int, set]        = defaultdict(set)

        # _confirmed_before[tid] = y2 lon nhat cua xe khi con truoc vach
        #   trong khi den do. Chi xe trong dict nay moi duoc xet vi pham.
        self._confirmed_before: dict[int, float]    = {}

        # _crossing_frame: frame luc xe vua cham vach (anh vi pham dep hon)
        self._crossing_frame: dict[int, np.ndarray] = {}

        self._plate_cache:    dict[int, str]        = {}

        # ── Light state ─────────────────────────────────────────────── #
        self._red_frame_count     = 0
        self._api_call_count      = 0
        self._cached_color        = "OFF"
        self._current_light_color = "OFF"
        self._prev_confirmed_red  = False
        self._cached_signs:  list = []
        self._api_error_count     = 0
        self._api_lock            = threading.Lock()
        self._api_thread: threading.Thread | None = None
        self._pending_api_call    = False

        # ── Test flag ────────────────────────────────────────────────── #
        self._force_red = False

        # ── Stop line ───────────────────────────────────────────────── #
        self._stop_line_y: int | None = None
        self._dragging_line           = False
        self._frame_counter           = 0

        self.violation_callback = None

        print(f"[DETECTOR v17] location='{location}' conf={conf} "
              f"force_red={self._force_red}")
        print(f"[DETECTOR v17] STOP_LINE_Y_RATIO={STOP_LINE_Y_RATIO} "
              f"RED_CONFIRM={RED_CONFIRM_FRAMES} "
              f"MIN_CROSSING_MOVE_PX={MIN_CROSSING_MOVE_PX}")

    # ── API async ────────────────────────────────────────────────────── #
    def _async_api_call(self, frame: np.ndarray):
        result = _call_traffic_api(frame, detect=["light", "sign"])
        color, source, signs_raw = _parse_api_response(result)
        with self._api_lock:
            if result:
                self._cached_color        = color
                self._current_light_color = color
                self._cached_signs        = signs_raw
                self._api_error_count     = 0
                print(f"[LIGHT API] color={color!r} src={source} "
                      f"signs={len(signs_raw)} call#{self._api_call_count}")
            else:
                self._api_error_count += 1
            self._pending_api_call = False

    def _trigger_api_async(self, frame: np.ndarray):
        with self._api_lock:
            if self._pending_api_call:
                return
            self._pending_api_call = True
        frame_copy = frame.copy()
        self._api_thread = threading.Thread(
            target=self._async_api_call, args=(frame_copy,), daemon=True)
        self._api_thread.start()

    # ── Stop line ────────────────────────────────────────────────────── #
    def _get_stop_y(self, frame_h: int) -> int:
        return (self._stop_line_y if self._stop_line_y is not None
                else int(frame_h * STOP_LINE_Y_RATIO))

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self._dragging_line = True
        elif event == cv2.EVENT_MOUSEMOVE and self._dragging_line:
            self._stop_line_y = y
        elif event == cv2.EVENT_LBUTTONUP:
            self._dragging_line = False
            print(f"[STOP LINE] y={self._stop_line_y}")

    # ── Geometry ─────────────────────────────────────────────────────── #
    @staticmethod
    def _center(bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @staticmethod
    def _distance(c1, c2):
        return ((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2) ** 0.5

    def _is_turning_right(self, tid):
        hist = self.track_history[tid]
        if len(hist) < MIN_TRACK_FRAMES:
            return False
        dx = hist[-1][0] - hist[-MIN_TRACK_FRAMES][0]
        dy = hist[-1][1] - hist[-MIN_TRACK_FRAMES][1]
        return dx > MIN_MOVE_PX and abs(dy) < MIN_MOVE_PX * 2

    def _is_turning_left(self, tid):
        hist = self.track_history[tid]
        if len(hist) < MIN_TRACK_FRAMES:
            return False
        dx = hist[-1][0] - hist[-MIN_TRACK_FRAMES][0]
        dy = hist[-1][1] - hist[-MIN_TRACK_FRAMES][1]
        return dx < -MIN_MOVE_PX and abs(dy) < MIN_MOVE_PX * 2

    def _is_entering_zone(self, tid):
        hist = self.track_history[tid]
        if len(hist) < MIN_TRACK_FRAMES:
            return False
        return (abs(hist[-1][0] - hist[-MIN_TRACK_FRAMES][0]) +
                abs(hist[-1][1] - hist[-MIN_TRACK_FRAMES][1])) > MIN_MOVE_PX

    def _is_wrong_way(self, tid):
        hist = self.track_history[tid]
        if len(hist) < MIN_TRACK_FRAMES:
            return False
        return (hist[-1][1] - hist[-MIN_TRACK_FRAMES][1]) < -MIN_MOVE_PX

    def _check_behavior(self, behavior, tid):
        if behavior == "turn_right": return self._is_turning_right(tid)
        if behavior == "turn_left":  return self._is_turning_left(tid)
        if behavior == "enter_zone": return self._is_entering_zone(tid)
        if behavior == "wrong_way":  return self._is_wrong_way(tid)
        return True

    def _update_red_state(self, raw_is_red: bool) -> bool:
        if raw_is_red:
            self._red_frame_count = min(self._red_frame_count + 1,
                                        RED_CONFIRM_FRAMES + 10)
        else:
            self._red_frame_count = max(0, self._red_frame_count - 1)
        return self._red_frame_count >= RED_CONFIRM_FRAMES

    # ── Save helpers ──────────────────────────────────────────────────── #
    def _save_vehicle_crop(self, frame, vehicle_bbox, ts_str, plate, vtype):
        fh, fw = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in vehicle_bbox]
        cx1 = max(0, x1 - CROP_PADDING)
        cy1 = max(0, y1 - CROP_PADDING)
        cx2 = min(fw, x2 + CROP_PADDING)
        cy2 = min(fh, y2 + CROP_PADDING)
        crop = frame[cy1:cy2, cx1:cx2].copy()
        if crop.size == 0:
            return None
        nw     = int(crop.shape[1] * CROP_ZOOM_FACTOR)
        nh     = int(crop.shape[0] * CROP_ZOOM_FACTOR)
        zoomed = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_CUBIC)
        ov = TextOverlay(zoomed)
        ov.add(f"VI PHAM: {vtype}", (8,  8),  FONT_MD, (0, 0, 255))
        ov.add(f"BSX: {plate}",     (8, 36),  FONT_SM, (0, 200, 255))
        ov.add(ts_str,               (8, 60),  FONT_SM, (200, 200, 200))
        zoomed = ov.render()
        path = os.path.join(IMG_DIR, f"{ts_str}_CROP_{plate}.jpg")
        cv2.imwrite(path, zoomed)
        return path

    def _save_violation(self, frame, vehicle_bbox, vehicle_class,
                        violation_type, sign_name, v_tid=-1):
        raw_plate = None
        if v_tid >= 0 and v_tid in self._plate_cache:
            raw_plate = self._plate_cache[v_tid]
        else:
            try:
                raw_plate = read_plate(frame, vehicle_bbox)
                if raw_plate and v_tid >= 0:
                    self._plate_cache[v_tid] = raw_plate
            except Exception as e:
                print(f"[LPR] Loi tid={v_tid}: {e}")

        if _is_valid_plate(raw_plate):
            plate       = _clean_plate(raw_plate)
            plate_clear = True
        else:
            plate       = "CHUA_RO"
            plate_clear = False

        ts_str   = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
        img_path = os.path.join(IMG_DIR, f"{ts_str}_{plate}.jpg")

        x1, y1, x2, y2 = [int(v) for v in vehicle_bbox]
        out = frame.copy()
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 3)
        ov = TextOverlay(out)
        ov.add(f"VI PHAM: {violation_type}", (x1, max(0, y1 - 32)),
               FONT_MD, (0, 0, 255))
        if plate_clear:
            ov.add(f"BSX: {plate}", (x1, y2 + 4), FONT_SM, (255, 255, 0))
        out = ov.render()
        cv2.imwrite(img_path, out)

        try:
            self._save_vehicle_crop(frame, vehicle_bbox, ts_str,
                                    plate, violation_type)
        except Exception as e:
            print(f"[CROP] Loi tid={v_tid}: {e}")

        vid = insert_violation(plate, vehicle_class, violation_type,
                               sign_name, img_path, self.location)
        if self.violation_callback:
            try:
                self.violation_callback(vid, plate, violation_type)
            except Exception as e:
                print(f"[CALLBACK] {e}")

        print(f"[VIOLATION] {'BSX:'+plate if plate_clear else 'CHUA RO'} "
              f"| {vehicle_class} | {violation_type}")
        return vid

    # ── Main process ─────────────────────────────────────────────────── #
    def process_frame(self, frame):
        new_violations = []
        frame_h, frame_w = frame.shape[:2]
        stop_y = self._get_stop_y(frame_h)

        # ── Vehicle tracking ─────────────────────────────────────────── #
        v_results = self.vehicle_model.track(
            frame, persist=True, conf=self.conf,
            tracker="bytetrack.yaml", verbose=False)

        vehicles = []
        if v_results and v_results[0].boxes is not None:
            v_names = self.vehicle_model.names
            for box in v_results[0].boxes:
                cls_id = int(box.cls[0])
                if cls_id not in self.vehicle_classes:
                    continue
                bbox = box.xyxy[0].tolist()
                tid  = int(box.id[0]) if box.id is not None else -1
                if tid >= 0:
                    cx, cy = self._center(bbox)
                    self.track_history[tid].append((cx, cy))
                    if len(self.track_history[tid]) > 60:
                        self.track_history[tid].pop(0)
                vehicles.append((bbox, v_names[cls_id], tid,
                                  float(box.conf[0])))

        # ── API call (light + sign) ───────────────────────────────────── #
        self._api_call_count += 1
        if (self._api_call_count % LIGHT_API_INTERVAL == 0
                or self._api_call_count == 1):
            self._trigger_api_async(frame)

        with self._api_lock:
            cached_color = self._cached_color
            cached_signs = list(self._cached_signs)

        # Parse signs tu API response
        signs = []
        for s in cached_signs:
            try:
                signs.append((s["bbox"], s["class"], s["conf"]))
            except (KeyError, TypeError):
                continue

        # ── Tinh trang thai den do ───────────────────────────────────── #
        raw_red = self._force_red or (cached_color == "RED")
        any_red = self._update_red_state(raw_red)

        # Debug log moi 30 frame
        if self._api_call_count % 30 == 0:
            print(f"[DEBUG] frame={self._api_call_count} "
                  f"color={cached_color!r} any_red={any_red} "
                  f"confirmed={len(self._confirmed_before)} "
                  f"vehicles={len(vehicles)} stop_y={stop_y}")

        # ── Logic 1: vi pham bien bao ────────────────────────────────── #
        for v_bbox, v_class, v_tid, _ in vehicles:
            v_center = self._center(v_bbox)
            for s_bbox, s_class, _ in signs:
                if self._distance(v_center, self._center(s_bbox)) > PROXIMITY_THRESHOLD:
                    continue
                rule = RULE_MAP.get(s_class)
                if not rule:
                    continue
                if rule["allowed"] and v_class in rule["allowed"]:
                    continue
                v_type = rule["violation_type"]
                if v_type in self.logged[v_tid]:
                    continue
                if not self._check_behavior(rule["check_behavior"], v_tid):
                    continue
                self.logged[v_tid].add(v_type)
                vid = self._save_violation(frame, v_bbox, v_class,
                                           v_type, s_class, v_tid)
                new_violations.append({"id": vid, "vehicle": v_class,
                                       "violation": v_type, "sign": s_class})

        # ── Logic 2: vuot den do ─────────────────────────────────────── #
        #
        # LUONG CHINH XAC:
        #
        # BUOC 1 - Ghi nhan xe truoc vach:
        #   Moi frame khi den do: xe nao co v_y2 < stop_y
        #   → luu _confirmed_before[tid] = max(y2 cu, y2 moi)
        #   (y2 lon nhat = xe o gan vach nhat khi con truoc vach)
        #
        # BUOC 2 - Kiem tra vi pham:
        #   Chi xet xe da co trong _confirmed_before
        #   Dieu kien VI PHAM: v_y2 >= stop_y
        #                  VA (v_y2 - _confirmed_before[tid]) >= MIN_CROSSING_MOVE_PX
        #   KHONG dung _is_moving_down — xe cho roi moi di: history y bang nhau
        #   → ham do luon tra False → chinh la bug khong bat duoc
        #
        # BUOC 3 - Reset khi den chuyen:
        #   Xoa _confirmed_before, _crossing_frame
        #   Xoa "Vuot den do" trong logged de bat duoc chu ky den tiep theo

        if not any_red:
            if self._prev_confirmed_red:
                # Den vua chuyen sang khong-do: xoa trang thai
                for tid in list(self._confirmed_before.keys()):
                    self.logged[tid].discard("Vuot den do")
                self._confirmed_before.clear()
                self._crossing_frame.clear()
                print("[LIGHT] Den chuyen khong-do -> reset trang thai vuot den")
        else:
            # BUOC 1: ghi nhan xe truoc vach
            # Them fallback: neu xe da qua vach nhung history cho thay
            # co diem truoc vach → confirm va tinh before_y tu history
            for v_bbox, v_class, v_tid, _ in vehicles:
                if v_tid < 0:
                    continue
                _, _, _, v_y2 = v_bbox
                if v_y2 < stop_y:
                    # Xe dang truoc vach: cap nhat y2 lon nhat
                    prev = self._confirmed_before.get(v_tid, 0.0)
                    self._confirmed_before[v_tid] = max(prev, v_y2)
                elif v_tid not in self._confirmed_before:
                    # Xe da qua vach nhung chua duoc confirm:
                    # Kiem tra history xem co tung o truoc vach khong
                    hist = self.track_history[v_tid]
                    before_pts = [p[1] for p in hist if p[1] < stop_y]
                    if before_pts:
                        # Co diem truoc vach trong history → confirm
                        self._confirmed_before[v_tid] = max(before_pts)
                        print(f"[CONFIRM via hist] tid={v_tid} "
                              f"before_y={self._confirmed_before[v_tid]:.0f}")

            # BUOC 2: kiem tra vi pham
            for v_bbox, v_class, v_tid, _ in vehicles:
                if v_tid < 0:
                    continue

                # Chi xet xe da duoc confirm truoc vach luc den do
                # (khong can MIN_TRACK_BEFORE_VIOLATION vi _confirmed_before da la guard)
                if v_tid not in self._confirmed_before:
                    continue

                if "Vuot den do" in self.logged[v_tid]:
                    continue

                x1, y1, x2, v_y2 = v_bbox

                # Xe van truoc vach → cap nhat y2 lon nhat
                if v_y2 < stop_y:
                    prev = self._confirmed_before.get(v_tid, 0.0)
                    self._confirmed_before[v_tid] = max(prev, v_y2)
                    continue

                # Xe da qua vach: tinh tong displacement tu vi tri truoc vach
                before_y   = self._confirmed_before[v_tid]
                moved_down = v_y2 - before_y

                # Log moi frame de debug (se xoa sau khi on dinh)
                print(f"[CHECK] tid={v_tid} cls={v_class} "
                      f"before={before_y:.0f} now={v_y2:.0f} "
                      f"moved={moved_down:.1f} stop_y={stop_y}")

                if moved_down < MIN_CROSSING_MOVE_PX:
                    # Chua du xa — bbox jitter hoac xe vua cham vach
                    continue

                # VI PHAM XAC NHAN
                self.logged[v_tid].add("Vuot den do")
                capture_frame = self._crossing_frame.get(v_tid, frame)
                vid = self._save_violation(
                    capture_frame, v_bbox, v_class,
                    "Vuot den do", "Den do", v_tid)
                new_violations.append(
                    {"id": vid, "vehicle": v_class,
                     "violation": "Vuot den do", "sign": "Den do"})
                print(f"[RED LIGHT ✓] tid={v_tid} cls={v_class} "
                      f"before_y={before_y:.0f} v_y2={v_y2:.0f} "
                      f"moved={moved_down:.0f}px STOP_Y={stop_y}")

            # Luu crossing_frame khi xe vua cham vach (anh dep hon)
            for v_bbox, v_class, v_tid, _ in vehicles:
                if v_tid < 0:
                    continue
                if v_tid not in self._confirmed_before:
                    continue
                if v_tid in self._crossing_frame:
                    continue
                if "Vuot den do" in self.logged[v_tid]:
                    continue
                x1, y1, x2, v_y2 = v_bbox
                if v_y2 >= stop_y and y1 < stop_y:
                    self._crossing_frame[v_tid] = frame.copy()

        self._prev_confirmed_red = any_red

        # ── Annotate ─────────────────────────────────────────────────── #
        annotated = (v_results[0].plot().copy()
                     if v_results and v_results[0].boxes is not None
                     else frame.copy())

        for s_bbox, s_class, _ in signs:
            bx1, by1, bx2, by2 = [int(v) for v in s_bbox]
            cv2.rectangle(annotated, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
            cv2.putText(annotated, s_class, (bx1, max(18, by1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1,
                        cv2.LINE_AA)

        # Stop line: do khi den do, xanh khi khong do
        line_color = (0, 0, 255) if any_red else (0, 200, 0)
        cv2.line(annotated, (0, stop_y), (frame_w, stop_y), line_color, 2)

        ov = TextOverlay(annotated)
        ov.add("VACH DUNG", (6, stop_y - 26), FONT_SM, line_color)

        _hud = {
            "RED":    ((0, 0, 255),     "DO"),
            "YELLOW": ((0, 200, 255),   "VANG"),
            "GREEN":  ((0, 200, 0),     "XANH"),
            "OFF":    ((120, 120, 120), "TAT"),
        }
        hc, ht = _hud.get(self._current_light_color, ((150, 150, 150), "?"))

        if self._force_red:
            ov.add("FORCE RED (TEST)", (frame_w - 260, 40), FONT_SM, (0, 0, 255))
        elif self._api_error_count >= 3:
            ov.add(f"API LOI: {self._api_error_count}x",
                   (frame_w - 200, 40), FONT_SM, (0, 100, 255))

        ov.add(f"DEN: {ht}", (frame_w - 160, 10), FONT_LG, hc)

        if any_red and self._confirmed_before:
            ov.add(f"Cho den do: {len(self._confirmed_before)}",
                   (6, stop_y + 6), FONT_SM, (0, 200, 255))

        for i, info in enumerate(new_violations):
            ov.add(f"VI PHAM: {info['violation']}",
                   (10, 10 + i * 30), FONT_LG, (0, 0, 255))

        annotated = ov.render()
        return annotated, new_violations

    # ── Run video ────────────────────────────────────────────────────── #
    def run_video(self, source=0, show=False, output_path=None):
        cap = cv2.VideoCapture(source)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            print(f"[ERROR] Khong mo duoc: {source}")
            return

        writer = None
        if output_path and isinstance(source, str):
            fps  = cap.get(cv2.CAP_PROP_FPS) or 25
            w, h = int(cap.get(3)), int(cap.get(4))
            writer = cv2.VideoWriter(output_path,
                                     cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        win_name = "Traffic Violation Detection"
        if show:
            try:
                cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
                cv2.setMouseCallback(win_name, self._mouse_callback)
                print("[INFO] Chuot TRAI: keo vach dung | Q: thoat")
            except cv2.error:
                show = False

        frame_count    = 0
        last_annotated = None
        t0             = time.time()
        self._api_call_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_count         += 1
            self._frame_counter += 1

            if self._frame_counter % FRAME_SKIP == 0:
                annotated, violations = self.process_frame(frame)
                last_annotated = annotated
            else:
                annotated = last_annotated if last_annotated is not None else frame

            if writer:
                writer.write(annotated)
            if show:
                try:
                    cv2.imshow(win_name, annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                except cv2.error:
                    show = False

        elapsed = time.time() - t0
        print(f"Hoan thanh! {frame_count} frames | "
              f"{frame_count/elapsed:.1f} FPS avg")
        cap.release()
        if writer:
            writer.release()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass


if __name__ == "__main__":
    detector = ViolationDetector(location="Camera - Nguyen Van Linh")
    detector.run_video(source="test_video.mp4", show=True)