# Hệ thống phát hiện vi phạm giao thông
**Đồ án chuyên ngành 1 — Khoa KHMT, VKUIT**
Nhóm: Lê Thị Kiều Loan · Võ Thị Kiều Dung · Trần Thị Thu Lam
GVHD: ThS. Lương Khánh Tý

---

## Cấu trúc project

```
traffic_system/
├── app.py          # Dashboard Streamlit (Thu Lam)
├── detection.py    # YOLOv8 + ByteTrack + Logic Engine (Kiều Loan)
├── lpr.py          # Nhận diện biển số EasyOCR (Kiều Dung)
├── database.py     # SQLite CRUD (Kiều Dung)
├── rules.py        # Bảng luật vi phạm (Kiều Loan)
├── requirements.txt
├── best.pt         # Model YOLOv8n đã fine-tune (đặt vào đây)
└── violations_img/ # Ảnh vi phạm tự động tạo
```

---

## Cài đặt

```bash
pip install -r requirements.txt
```

---

## Chạy hệ thống

**Terminal 1 — chạy detection engine:**
```bash
python detection.py
```

**Terminal 2 — mở dashboard:**
```bash
streamlit run app.py
```

---

## 5 loại vi phạm phát hiện

| Biển báo | Vi phạm |
|---|---|
| P.106b | Ô tô rẽ phải khi chỉ xe máy được phép |
| P.130  | Đi vào đường cấm |
| R.412  | Đi ngược chiều |
| P.102  | Ô tô vào đường cấm ô tô |
| P.104  | Xe tải vào đường cấm tải |

---

## Tính năng dashboard

- Tổng quan KPI + biểu đồ xu hướng
- Danh sách vi phạm: tìm kiếm, lọc, xem ảnh bằng chứng
- Xác nhận / xóa từng record
- Camera trực tiếp với realtime detection
- Thống kê theo giờ, theo thứ, theo loại biển báo
- Xuất báo cáo CSV
