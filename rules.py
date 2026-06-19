# rules.py — v2
# Class names theo model custom (tên tiếng Việt)

# ── Vehicle classes ───────────────────────────────────────────────────── #
# ID sẽ được build động trong detection.py dựa theo model.names
# Tên phải khớp chính xác với output của model xe (print(model.names))
VEHICLE_CLASS_NAMES = {"xe buýt", "xe tải", "xe máy", "ô tô"}

# Set này được điền tự động bởi ViolationDetector.__init__
VEHICLE_CLASSES: set[int] = set()

# ── Ngưỡng ────────────────────────────────────────────────────────────── #
PROXIMITY_THRESHOLD = 200.0
MIN_TRACK_FRAMES    = 5
MIN_MOVE_PX         = 10

# ── Rule map — key phải khớp class name trong sign model ─────────────── #
# Sau khi chạy traffic_api.py, xem log "[SIGN MODEL] Classes: {...}"
# rồi điều chỉnh key bên dưới cho đúng
RULE_MAP = {
    "cam_di_nguoc_chieu": {
        "violation_type": "Di nguoc chieu",
        "allowed": [],
        "check_behavior": "wrong_way",
    },
    "cam_o_to": {
        "violation_type": "Duong cam o to",
        "allowed": ["xe máy"],
        "check_behavior": "enter_zone",
    },
    "cam_xe_may": {
        "violation_type": "Duong cam xe may",
        "allowed": ["ô tô", "xe tải", "xe buýt"],
        "check_behavior": "enter_zone",
    },
    "cam_re_trai": {
        "violation_type": "Re trai cam",
        "allowed": [],
        "check_behavior": "turn_left",
    },
    "cam_re_phai": {
        "violation_type": "Re phai cam",
        "allowed": [],
        "check_behavior": "turn_right",
    },
}