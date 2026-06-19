import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import threading
import queue
from datetime import datetime, date, timedelta
import cv2
import time
from database import (get_all, get_stats, delete_violation,
                      confirm_violation, export_csv, init_db)

st.set_page_config(
    page_title="Traffic Violation · Giám sát giao thông",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)
init_db()

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #0c0e16; }
section[data-testid="stSidebar"] {
    background: #111320; border-right: 1px solid #1e2235; width: 220px !important;
}
section[data-testid="stSidebar"] * { color: #cbd5e1 !important; }
section[data-testid="stSidebar"] .stRadio label {
    display: flex; align-items: center; gap: 8px; padding: 9px 14px;
    border-radius: 8px; font-size: 13px; cursor: pointer; transition: background 0.15s;
}
section[data-testid="stSidebar"] .stRadio label:hover { background: #1e2235; }
.kpi-wrap {
    background: #151828; border: 1px solid #1e2235; border-radius: 14px;
    padding: 18px 20px 14px; position: relative; overflow: hidden;
}
.kpi-accent { position: absolute; top: 0; left: 0; right: 0; height: 3px; border-radius: 14px 14px 0 0; }
.kpi-label { font-size: 11px; font-weight: 500; letter-spacing: 0.06em; text-transform: uppercase; color: #64748b; margin-bottom: 10px; }
.kpi-value { font-size: 28px; font-weight: 600; line-height: 1; margin-bottom: 6px; }
.kpi-sub { font-size: 11px; color: #475569; }
.chart-card { background: #151828; border: 1px solid #1e2235; border-radius: 14px; padding: 18px 20px; margin-bottom: 16px; }
.chart-title { font-size: 13px; font-weight: 600; color: #e2e8f0; margin-bottom: 4px; }
.chart-sub { font-size: 11px; color: #64748b; margin-bottom: 14px; }
.badge { display: inline-block; padding: 2px 9px; border-radius: 20px; font-size: 11px; font-weight: 600; }
.badge-red    { background: rgba(239,68,68,.15);  color: #fca5a5; }
.badge-green  { background: rgba(34,197,94,.15);  color: #86efac; }
.badge-amber  { background: rgba(234,179,8,.15);  color: #fde68a; }
.badge-blue   { background: rgba(59,130,246,.15); color: #93c5fd; }
.badge-purple { background: rgba(139,92,246,.15); color: #c4b5fd; }
.status-bar { display: flex; align-items: center; gap: 6px; padding: 5px 12px; background: #151828; border: 1px solid #1e2235; border-radius: 20px; font-size: 12px; color: #94a3b8; }
.dot { width: 7px; height: 7px; border-radius: 50%; }
.dot-green { background: #22c55e; }
.dot-red   { background: #ef4444; animation: pulse 1.5s infinite; }
@keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }
.sec-title { font-size: 12px; font-weight: 600; letter-spacing: 0.07em; text-transform: uppercase; color: #475569; margin-bottom: 10px; padding-bottom: 8px; border-bottom: 1px solid #1e2235; }
hr { border-color: #1e2235 !important; }
#MainMenu, footer, header { visibility: hidden; }
div[data-testid="stDecoration"] { display: none; }
div.stButton > button { border-radius: 8px; font-size: 13px; font-weight: 500; border: 1px solid #2d3659; background: #1e2235; color: #e2e8f0; transition: all 0.15s; }
div.stButton > button:hover { border-color: #ef4444; color: #ef4444; background: rgba(239,68,68,.08); }
div.stButton > button[kind="primary"] { background: #ef4444; border-color: #ef4444; color: #fff; }
.stSlider > div > div { accent-color: #ef4444; }
.stTextInput input { background: #151828 !important; border-color: #1e2235 !important; color: #e2e8f0 !important; border-radius: 8px !important; }
</style>
""", unsafe_allow_html=True)

PLOT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", color="#94a3b8", size=11),
    margin=dict(l=8, r=8, t=28, b=8), height=240,
    showlegend=False,
    xaxis=dict(gridcolor="#1e2235", linecolor="#1e2235", tickfont_color="#64748b"),
    yaxis=dict(gridcolor="#1e2235", linecolor="#1e2235", tickfont_color="#64748b"),
)
COLORS = ["#ef4444", "#f97316", "#eab308", "#22c55e", "#3b82f6", "#8b5cf6"]

with st.sidebar:
    st.markdown("""
    <div style='padding:16px 0 20px;'>
        <div style='font-size:24px;margin-bottom:6px'>🚦</div>
        <div style='font-size:14px;font-weight:600;color:#f1f5f9'>Traffic Violation</div>
        <div style='font-size:11px;color:#475569;margin-top:2px'>Hệ thống giám sát giao thông</div>
    </div>
    """, unsafe_allow_html=True)
    page = st.radio("nav",
                    ["📊  Tổng quan", "📋  Danh sách vi phạm",
                     "📸  Camera trực tiếp", "📈  Thống kê"],
                    label_visibility="collapsed")
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:11px;color:#334155;line-height:2'>
        👥 Nhóm: Loan · Dung · Lam<br>
        🎓 GVHD: ThS. Lương Khánh Tý<br>
        📅 HK2 – 2025/2026
    </div>""", unsafe_allow_html=True)


def kpi_card(label, value, sub="", accent="#ef4444"):
    st.markdown(f"""
    <div class='kpi-wrap'>
        <div class='kpi-accent' style='background:{accent}'></div>
        <div class='kpi-label'>{label}</div>
        <div class='kpi-value' style='color:{accent}'>{value}</div>
        <div class='kpi-sub'>{sub}</div>
    </div>""", unsafe_allow_html=True)


def badge(text, color="red"):
    return f"<span class='badge badge-{color}'>{text}</span>"


def section(title):
    st.markdown(f"<div class='sec-title'>{title}</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════ #
#  TRANG 1 — TỔNG QUAN
# ══════════════════════════════════════════════════════════════════════ #
if page == "📊  Tổng quan":
    c_h1, c_h2 = st.columns([3, 1])
    c_h1.markdown("<h2 style='color:#f1f5f9;font-size:20px;font-weight:600;margin-bottom:4px'>📊 Tổng quan</h2>", unsafe_allow_html=True)
    with c_h2:
        st.markdown("<div style='display:flex;justify-content:flex-end;gap:8px;margin-top:4px'><div class='status-bar'><span class='dot dot-green'></span>Hệ thống hoạt động</div></div>", unsafe_allow_html=True)

    stats  = get_stats()
    df_all = get_all()

    k1, k2, k3, k4 = st.columns(4)
    with k1: kpi_card("Tổng vi phạm",  stats["total"],     "Tất cả thời gian", "#ef4444")
    with k2: kpi_card("Hôm nay",       stats["today"],     date.today().strftime("%d/%m/%Y"), "#f97316")
    with k3: kpi_card("Đã xác nhận",   stats["confirmed"], f"{int(stats['confirmed']/max(1,stats['total'])*100) if stats['total'] > 0 else 0}% tổng số", "#22c55e")
    with k4: kpi_card("Chờ xử lý",     stats["total"] - stats["confirmed"], "Cần xem xét", "#3b82f6")

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    if df_all.empty:
        st.markdown("<div style='text-align:center;padding:4rem;background:#151828;border-radius:14px;border:1px dashed #1e2235'><div style='font-size:3rem;margin-bottom:8px'>📭</div><div style='color:#64748b;font-size:14px'>Chưa có dữ liệu vi phạm</div></div>", unsafe_allow_html=True)
    else:
        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
            st.markdown("<div class='chart-title'>Vi phạm theo loại xe</div><div class='chart-sub'>Phân bổ phương tiện</div>", unsafe_allow_html=True)
            vc = df_all["vehicle_type"].value_counts().reset_index(); vc.columns = ["Loại xe","Số lần"]
            fig = px.bar(vc, x="Loại xe", y="Số lần", color="Loại xe", color_discrete_sequence=COLORS)
            fig.update_layout(**PLOT); fig.update_traces(marker_line_width=0, marker_cornerradius=4)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            st.markdown("</div>", unsafe_allow_html=True)
        with col_r:
            st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
            st.markdown("<div class='chart-title'>Phân bố loại vi phạm</div><div class='chart-sub'>Tỉ lệ từng hành vi</div>", unsafe_allow_html=True)
            viol = df_all["violation_type"].value_counts().reset_index(); viol.columns = ["Vi phạm","Số lần"]
            fig2 = px.pie(viol, names="Vi phạm", values="Số lần", hole=0.52, color_discrete_sequence=COLORS)
            fig2.update_layout(**PLOT); fig2.update_traces(textfont_color="#e2e8f0", marker_line_color="rgba(0,0,0,0)")
            st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
        st.markdown("<div class='chart-title'>Xu hướng vi phạm 30 ngày</div><div class='chart-sub'>Số lượng mỗi ngày</div>", unsafe_allow_html=True)
        df_all["date"] = pd.to_datetime(df_all["timestamp"]).dt.date
        trend = df_all.groupby("date").size().reset_index(name="count")
        fig3 = px.area(trend, x="date", y="count", labels={"date":"Ngày","count":"Số vi phạm"}, color_discrete_sequence=["#ef4444"])
        fig3.update_traces(fill="tozeroy", fillcolor="rgba(239,68,68,0.12)", line_color="#ef4444", line_width=2)
        fig3.update_layout(**{**PLOT, "height": 200})
        st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

        section("🕐 5 vi phạm mới nhất")
        recent = (df_all.head(5)[["timestamp","plate","vehicle_type","violation_type","location"]]
                  .rename(columns={"timestamp":"Thời gian","plate":"Biển số","vehicle_type":"Loại xe","violation_type":"Vi phạm","location":"Vị trí"}))
        st.dataframe(recent, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════ #
#  TRANG 2 — DANH SÁCH VI PHẠM
# ══════════════════════════════════════════════════════════════════════ #
elif page == "📋  Danh sách vi phạm":
    st.markdown("<h2 style='color:#f1f5f9;font-size:20px;font-weight:600;margin-bottom:16px'>📋 Danh sách vi phạm</h2>", unsafe_allow_html=True)

    with st.expander("🔍 Bộ lọc tìm kiếm", expanded=True):
        cf1, cf2, cf3, cf4 = st.columns(4)
        search = cf1.text_input("Biển số / Vi phạm / Vị trí", placeholder="Nhập từ khóa…")
        vtype  = cf2.selectbox("Loại xe", ["Tất cả","car","motorbike","truck","bus"])
        d_from = cf3.date_input("Từ ngày", value=date.today() - timedelta(days=30))
        d_to   = cf4.date_input("Đến ngày", value=date.today())

    df = get_all(search=search, vtype=vtype, date_from=d_from, date_to=d_to)
    col_info, col_btn = st.columns([3,1])
    col_info.markdown(f"<div style='color:#64748b;font-size:13px;margin-top:8px'>Tìm thấy <b style='color:#f1f5f9'>{len(df)}</b> kết quả</div>", unsafe_allow_html=True)
    if not df.empty:
        csv_path = export_csv(df)
        with open(csv_path, "rb") as f:
            col_btn.download_button("⬇ Xuất CSV", data=f, file_name="bao_cao_vi_pham.csv", mime="text/csv", use_container_width=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    if df.empty:
        st.markdown("<div style='text-align:center;padding:3rem;background:#151828;border-radius:14px;border:1px dashed #1e2235;color:#475569'>Không tìm thấy vi phạm nào</div>", unsafe_allow_html=True)
    else:
        for _, row in df.iterrows():
            confirmed  = bool(row["confirmed"])
            border_clr = "#22c55e" if confirmed else "#ef4444"
            status_html = badge("✅ Đã xác nhận","green") if confirmed else badge("⏳ Chờ xử lý","red")
            vtype_color = {"car":"blue","motorbike":"amber","truck":"purple","bus":"green"}.get(row["vehicle_type"],"blue")

            with st.container():
                st.markdown(f"<div style='border-left:3px solid {border_clr};background:#151828;border-radius:12px;border:1px solid #1e2235;padding:14px 18px;margin-bottom:10px'>", unsafe_allow_html=True)
                r1, r2, r3 = st.columns([1,3,1])
                with r1:
                    if row["image_path"] and os.path.exists(str(row["image_path"])):
                        st.image(row["image_path"], use_container_width=True, caption="")
                    else:
                        st.markdown("<div style='background:#1e2235;border-radius:8px;padding:2rem 1rem;text-align:center;color:#334155;font-size:20px'>📷</div>", unsafe_allow_html=True)
                with r2:
                    st.markdown(f"""
                    <div style='color:#f1f5f9;font-size:14px;font-weight:600;display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:6px'>
                        <span style='color:#475569;font-size:11px'>#{row['id']}</span>
                        🪪 {row['plate']} {status_html} {badge(row['vehicle_type'], vtype_color)}
                    </div>
                    <div style='font-size:13px;color:#94a3b8;line-height:2.0'>
                        ⚠️ <b style='color:#fca5a5'>{row['violation_type']}</b><br>
                        🪧 Biển: <span style='color:#e2e8f0'>{row['sign_detected']}</span> | 📍 {row['location']}<br>
                        🕐 <span style='font-size:12px;color:#64748b'>{row['timestamp']}</span>
                    </div>""", unsafe_allow_html=True)
                with r3:
                    if not confirmed:
                        if st.button("✅ Xác nhận", key=f"ok_{row['id']}", use_container_width=True):
                            confirm_violation(row["id"]); st.rerun()
                    if st.button("🗑 Xóa", key=f"del_{row['id']}", use_container_width=True):
                        delete_violation(row["id"]); st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════ #
#  TRANG 3 — CAMERA TRỰC TIẾP
# ══════════════════════════════════════════════════════════════════════ #
elif page == "📸  Camera trực tiếp":
    st.markdown("<h2 style='color:#f1f5f9;font-size:20px;font-weight:600;margin-bottom:16px'>📸 Camera trực tiếp</h2>", unsafe_allow_html=True)

    # ── Khởi tạo session_state ──
    for key, default in [
        ("cam_running",   False),
        ("detector",      None),
        ("cap",           None),
        ("violations",    []),
        ("frame_count",   0),
        ("t0",            time.time()),
        ("fps_display",   0.0),
        # Thread-safe frame queue: maxsize=1 giữ frame mới nhất
        ("frame_queue",   None),
        ("stop_event",    None),
        ("cam_thread",    None),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    col_a, col_b = st.columns([1, 2], gap="medium")

    with col_a:
        section("⚙️ Cấu hình nguồn")
        source_type = st.radio("Nguồn video", ["Webcam","File video","IP Camera"], horizontal=True)
        source = None
        if source_type == "Webcam":
            source = int(st.number_input("Camera ID", value=0, min_value=0))
        elif source_type == "File video":
            uploaded = st.file_uploader("Tải lên video", type=["mp4","avi","mov"])
            if uploaded:
                tmp_path = "temp_upload.mp4"
                with open(tmp_path, "wb") as f:
                    f.write(uploaded.read())
                source = tmp_path
        else:
            source = st.text_input("URL camera (RTSP)", placeholder="rtsp://…")

        location  = st.text_input("📍 Tên vị trí camera", value="Camera Nguyễn Văn Linh")
        conf_thr  = st.slider("🎯 Ngưỡng confidence", 0.10, 0.90, 0.35, 0.05)
        display_w = st.slider("🖥 Độ rộng hiển thị (px)", 480, 1280, 720, 80)

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        col_btn1, col_btn2 = st.columns(2)
        start_btn = col_btn1.button("▶ Bắt đầu", type="primary", use_container_width=True)
        stop_btn  = col_btn2.button("⏹ Dừng",                   use_container_width=True)

        section("🚨 Vi phạm vừa phát hiện")
        violation_log   = st.empty()
        fps_placeholder = st.empty()

    with col_b:
        section("🎥 Luồng video")
        frame_placeholder = st.empty()

    # ── Hàm chạy detection trong thread riêng ──
    def _camera_worker(source, detector, stop_event, frame_queue, violation_list):
        """
        Chạy trong thread nền: đọc frame → detect → đẩy vào queue.
        Không đụng vào Streamlit API — hoàn toàn thread-safe.
        """
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            stop_event.set()
            return

        frame_count = 0
        last_annot  = None
        t0 = time.time()

        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                stop_event.set()
                break

            frame_count += 1

            # FRAME_SKIP = 2 để giảm tải CPU
            if frame_count % 2 == 0:
                try:
                    annotated, _ = detector.process_frame(frame)
                    last_annot = annotated
                except Exception:
                    annotated = frame
            else:
                annotated = last_annot if last_annot is not None else frame

            # Resize trước khi đẩy vào queue
            h, w = annotated.shape[:2]
            if w > 720:  # dùng 720 mặc định trong thread; resize lại ngoài UI nếu cần
                scale = 720 / w
                annotated = cv2.resize(annotated, (720, int(h * scale)), interpolation=cv2.INTER_AREA)

            elapsed = time.time() - t0
            fps     = frame_count / elapsed if elapsed > 0 else 0.0

            # Đẩy vào queue — nếu queue đầy thì bỏ frame cũ
            payload = (annotated, fps, frame_count, list(violation_list))
            try:
                frame_queue.get_nowait()  # xoá frame cũ chưa kịp render
            except Exception:
                pass
            try:
                frame_queue.put_nowait(payload)
            except Exception:
                pass

            time.sleep(0.01)  # ~100 FPS tối đa, thực tế bị giới hạn bởi model

        cap.release()

    # ── Nút BẮT ĐẦU ──
    if start_btn and source is not None and not st.session_state.cam_running:
        # Dừng thread cũ nếu còn
        if st.session_state.stop_event is not None:
            st.session_state.stop_event.set()
        if st.session_state.cam_thread is not None and st.session_state.cam_thread.is_alive():
            st.session_state.cam_thread.join(timeout=2)

        try:
            from detection import ViolationDetector

            detector = ViolationDetector(location=location, conf=conf_thr)

            # Violation callback vẫn chạy trong thread → ghi vào list shared
            violations_shared = []

            def on_violation(vid, plate, vtype):
                ts = datetime.now().strftime("%H:%M:%S")
                violations_shared.insert(0, (ts, plate, vtype))
                if len(violations_shared) > 20:
                    violations_shared.pop()

            detector.violation_callback = on_violation

            stop_event  = threading.Event()
            frame_queue = queue.Queue(maxsize=1)

            cam_thread = threading.Thread(
                target=_camera_worker,
                args=(source, detector, stop_event, frame_queue, violations_shared),
                daemon=True,
            )
            cam_thread.start()

            st.session_state.cam_running      = True
            st.session_state.detector         = detector
            st.session_state.violations       = violations_shared
            st.session_state.stop_event       = stop_event
            st.session_state.frame_queue      = frame_queue
            st.session_state.cam_thread       = cam_thread
            st.session_state.frame_count      = 0
            st.session_state.t0               = time.time()
            st.session_state.fps_display      = 0.0

        except Exception as e:
            st.error(f"❌ Lỗi khởi chạy nguồn video: {e}")

    # ── Nút DỪNG ──
    if stop_btn and st.session_state.cam_running:
        if st.session_state.stop_event is not None:
            st.session_state.stop_event.set()
        st.session_state.cam_running = False
        st.success("✅ Đã dừng camera!")

    # ── Vòng lặp hiển thị — chỉ cập nhật placeholder, KHÔNG rerun toàn trang ──
    if st.session_state.cam_running:
        fq = st.session_state.frame_queue
        se = st.session_state.stop_event

        # Nếu thread worker đã kết thúc
        if se is not None and se.is_set():
            st.session_state.cam_running = False
            st.info("✅ Luồng video đã kết thúc.")
        else:
            # while loop: render liên tục KHÔNG dùng st.rerun()
            while st.session_state.cam_running:
                # Kiểm tra thread có còn sống không
                thread = st.session_state.cam_thread
                if thread is not None and not thread.is_alive():
                    st.session_state.cam_running = False
                    st.info("✅ Luồng video đã kết thúc.")
                    break

                # Lấy frame mới nhất từ queue
                try:
                    annotated, fps, fc, viols = fq.get(timeout=0.5)
                except queue.Empty:
                    # Chưa có frame mới, thử lại
                    continue

                # Resize theo display_w người dùng chọn
                h, w = annotated.shape[:2]
                if w != display_w:
                    scale     = display_w / w
                    annotated = cv2.resize(
                        annotated,
                        (display_w, int(h * scale)),
                        interpolation=cv2.INTER_AREA,
                    )

                # Cập nhật chỉ đúng placeholder — không render lại toàn trang
                frame_placeholder.image(annotated, channels="BGR")

                fps_placeholder.markdown(
                    f"🚀 Tốc độ: **{fps:.1f} FPS** | Frames: **{fc}**"
                )

                if viols:
                    with violation_log.container():
                        st.markdown(
                            "<div style='max-height:180px;overflow-y:auto;'>",
                            unsafe_allow_html=True,
                        )
                        for ts, plate, vtype in viols[:6]:
                            st.markdown(f"⏱️ `{ts}` | 🪪 `{plate}` | ⚠️ **{vtype}**")
                        st.markdown("</div>", unsafe_allow_html=True)

                # Giới hạn tốc độ render UI ~30 FPS để tránh quá tải browser
                time.sleep(1 / 30)


# ══════════════════════════════════════════════════════════════════════ #
#  TRANG 4 — THỐNG KÊ CHI TIẾT
# ══════════════════════════════════════════════════════════════════════ #
elif page == "📈  Thống kê":
    st.markdown("<h2 style='color:#f1f5f9;font-size:20px;font-weight:600;margin-bottom:16px'>📈 Thống kê & Phân tích chuyên sâu</h2>", unsafe_allow_html=True)

    df_all = get_all()

    if df_all.empty:
        st.markdown("<div style='text-align:center;padding:4rem;background:#151828;border-radius:14px;border:1px dashed #1e2235'><div style='font-size:3rem;margin-bottom:8px'>📭</div><div style='color:#64748b;font-size:14px'>Chưa có dữ liệu vi phạm để thống kê</div></div>", unsafe_allow_html=True)
    else:
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)

        total_count     = len(df_all)
        confirmed_count = len(df_all[df_all["confirmed"] == 1])
        pending_count   = total_count - confirmed_count
        ratio           = (confirmed_count / total_count * 100) if total_count > 0 else 0

        with m1: kpi_card("Tỷ lệ xác thực",    f"{ratio:.1f}%",                      "Độ chính xác giám sát",         "#22c55e")
        with m2: kpi_card("Chờ xử lý",          str(pending_count),                   "Hồ sơ vi phạm chưa duyệt",      "#3b82f6")
        with m3: kpi_card("Địa điểm ghi nhận",  str(len(df_all["location"].unique())), "Số lượng camera hoạt động",     "#8b5cf6")
        with m4: kpi_card("Loại vi phạm",        str(len(df_all["violation_type"].unique())), "Các dạng hành vi vi phạm", "#f97316")

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

        c1, c2 = st.columns(2)

        with c1:
            st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
            st.markdown("<div class='chart-title'>Khung giờ phát sinh vi phạm</div><div class='chart-sub'>Thống kê số lượng theo từng giờ trong ngày</div>", unsafe_allow_html=True)

            df_all["hour"]  = pd.to_datetime(df_all["timestamp"]).dt.hour
            hour_counts     = df_all["hour"].value_counts().reindex(range(24), fill_value=0).reset_index()
            hour_counts.columns = ["Giờ", "Số vụ"]

            fig_hour = px.line(hour_counts, x="Giờ", y="Số vụ", color_discrete_sequence=["#ef4444"])
            fig_hour.update_layout(**PLOT)
            fig_hour.update_traces(line_width=3, marker=dict(size=6, color="#ef4444"))
            st.plotly_chart(fig_hour, use_container_width=True, config={"displayModeBar": False})
            st.markdown("</div>", unsafe_allow_html=True)

        with c2:
            st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
            st.markdown("<div class='chart-title'>Thống kê theo vị trí Camera</div><div class='chart-sub'>Các điểm nóng giao thông phát sinh vi phạm</div>", unsafe_allow_html=True)

            loc_counts = df_all["location"].value_counts().reset_index()
            loc_counts.columns = ["Địa điểm", "Số vụ"]

            fig_loc = px.bar(loc_counts, y="Địa điểm", x="Số vụ", orientation="h", color="Địa điểm", color_discrete_sequence=COLORS)
            fig_loc.update_layout(**PLOT)
            fig_loc.update_traces(marker_line_width=0, marker_cornerradius=4)
            st.plotly_chart(fig_loc, use_container_width=True, config={"displayModeBar": False})
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
        st.markdown("<div class='chart-title'>Ma trận hành vi vi phạm & Phương tiện</div><div class='chart-sub'>Chi tiết phân loại vi phạm của từng loại xe</div>", unsafe_allow_html=True)

        pivot_df = df_all.groupby(["vehicle_type", "violation_type"]).size().reset_index(name="counts")

        fig_matrix = px.bar(
            pivot_df, x="vehicle_type", y="counts", color="violation_type",
            barmode="group",
            labels={"vehicle_type": "Loại xe", "counts": "Số vụ", "violation_type": "Hành vi"},
            color_discrete_sequence=COLORS,
        )
        fig_matrix.update_layout(**{**PLOT, "height": 280, "showlegend": True})
        fig_matrix.update_traces(marker_line_width=0, marker_cornerradius=4)
        st.plotly_chart(fig_matrix, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)