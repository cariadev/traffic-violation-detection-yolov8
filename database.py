import sqlite3
import pandas as pd
import os
from datetime import datetime
# Đường dẫn tương đối để lưu trong thư mục dự án
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "traffic_violations.db")
IMG_DIR = os.path.join(BASE_DIR, "violations_images")
# Đảm bảo thư mục lưu ảnh vi phạm được tạo
os.makedirs(IMG_DIR, exist_ok=True)
def init_db():
    """Khởi tạo cơ sở dữ liệu SQLite và bảng vi phạm nếu chưa tồn tại"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS violations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            plate TEXT NOT NULL,
            vehicle_type TEXT NOT NULL,
            violation_type TEXT NOT NULL,
            sign_detected TEXT NOT NULL,
            image_path TEXT NOT NULL,
            location TEXT NOT NULL,
            confirmed INTEGER DEFAULT 0
        )
    """)
    conn.commit()
    conn.close()
def insert_violation(plate, vehicle_class, violation_type, sign_name, img_path, location):
    """Thêm một bản ghi vi phạm mới vào CSDL"""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("""
        INSERT INTO violations (timestamp, plate, vehicle_type, violation_type, sign_detected, image_path, location, confirmed)
        VALUES (?, ?, ?, ?, ?, ?, ?, 0)
    """, (timestamp, plate, vehicle_class, violation_type, sign_name, img_path, location))
    vid = cursor.lastrowid
    conn.commit()
    conn.close()
    return vid
def get_all(search=None, vtype=None, date_from=None, date_to=None):
    """Truy vấn tất cả vi phạm kết hợp bộ lọc tìm kiếm nâng cao"""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    
    query = "SELECT * FROM violations WHERE 1=1"
    params = []
    
    if search:
        query += " AND (plate LIKE ? OR violation_type LIKE ? OR location LIKE ?)"
        search_val = f"%{search}%"
        params.extend([search_val, search_val, search_val])
        
    if vtype and vtype != "Tất cả":
        query += " AND vehicle_type = ?"
        params.append(vtype)
        
    if date_from:
        query += " AND date(timestamp) >= date(?)"
        params.append(str(date_from))
        
    if date_to:
        query += " AND date(timestamp) <= date(?)"
        params.append(str(date_to))
        
    query += " ORDER BY timestamp DESC"
    
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df
def get_stats():
    """Lấy số liệu thống kê cơ bản hiển thị lên KPI Dashboard"""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Tổng vi phạm
    cursor.execute("SELECT COUNT(*) FROM violations")
    total = cursor.fetchone()[0]
    
    # Vi phạm hôm nay
    today_str = datetime.now().strftime("%Y-%m-%d")
    cursor.execute("SELECT COUNT(*) FROM violations WHERE date(timestamp) = date(?)", (today_str,))
    today = cursor.fetchone()[0]
    
    # Vi phạm đã xác nhận
    cursor.execute("SELECT COUNT(*) FROM violations WHERE confirmed = 1")
    confirmed = cursor.fetchone()[0]
    
    conn.close()
    
    return {
        "total": total,
        "today": today,
        "confirmed": confirmed
    }
def delete_violation(vid):
    """Xóa một bản ghi vi phạm và tệp ảnh chụp đi kèm"""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("SELECT image_path FROM violations WHERE id = ?", (vid,))
    row = cursor.fetchone()
    if row:
        img_path = row[0]
        # Xóa tệp ảnh nếu tồn tại
        if img_path and os.path.exists(img_path):
            try:
                os.remove(img_path)
                # Thử xóa ảnh crop liên quan
                crop_path = img_path.replace(".jpg", "_crop.jpg")
                if os.path.exists(crop_path):
                    os.remove(crop_path)
            except Exception as e:
                print(f"[DB] Không thể xóa tệp ảnh: {e}")
                
    cursor.execute("DELETE FROM violations WHERE id = ?", (vid,))
    conn.commit()
    conn.close()
def confirm_violation(vid):
    """Cập nhật trạng thái đã xác nhận vi phạm"""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("UPDATE violations SET confirmed = 1 WHERE id = ?", (vid,))
    conn.commit()
    conn.close()
def export_csv(df):
    """Xuất DataFrame vi phạm sang tệp CSV báo cáo tiếng Việt"""
    csv_path = os.path.join(BASE_DIR, "bao_cao_vi_pham.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    return csv_path