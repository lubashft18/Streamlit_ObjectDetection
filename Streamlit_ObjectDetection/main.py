
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
from ultralytics import YOLO
import av
import cv2
import numpy as np
import threading
import time
from datetime import datetime
from collections import defaultdict, deque
import os

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Object Tracker Pro",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS (unchanged) ───────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Rajdhani', sans-serif;
    background-color: #0a0e1a;
    color: #e0e8ff;
}

.stApp { background: #0a0e1a; }

h1, h2, h3 { font-family: 'Share Tech Mono', monospace; color: #00f5d4; }

.metric-card {
    background: linear-gradient(135deg, #0d1b2a 0%, #1a2744 100%);
    border: 1px solid #00f5d420;
    border-radius: 12px;
    padding: 16px;
    text-align: center;
    box-shadow: 0 0 20px #00f5d410;
}

.metric-value {
    font-family: 'Share Tech Mono', monospace;
    font-size: 2rem;
    color: #00f5d4;
    line-height: 1;
}

.metric-label {
    font-size: 0.8rem;
    color: #7a8aaa;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-top: 4px;
}

.alert-box {
    background: linear-gradient(135deg, #2a0d0d, #440f0f);
    border: 1px solid #ff4444;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 6px 0;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.85rem;
    color: #ff8888;
    animation: pulse 1.5s infinite;
}

@keyframes pulse {
    0%, 100% { border-color: #ff4444; box-shadow: 0 0 8px #ff444430; }
    50% { border-color: #ff8888; box-shadow: 0 0 16px #ff444460; }
}

.object-pill {
    display: inline-block;
    background: #00f5d415;
    border: 1px solid #00f5d440;
    border-radius: 20px;
    padding: 4px 12px;
    margin: 3px;
    font-size: 0.8rem;
    color: #00f5d4;
    font-family: 'Share Tech Mono', monospace;
}

.stButton > button {
    background: linear-gradient(135deg, #00f5d4, #0090ff);
    color: #0a0e1a;
    border: none;
    border-radius: 8px;
    font-family: 'Share Tech Mono', monospace;
    font-weight: 700;
    letter-spacing: 1px;
    padding: 8px 20px;
    transition: all 0.2s;
}

.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 20px #00f5d450;
}

.sidebar .stSelectbox label, .sidebar .stSlider label, .sidebar .stMultiSelect label {
    color: #7a8aaa !important;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 1px;
}

div[data-testid="stSidebar"] {
    background: #0d1320;
    border-right: 1px solid #1a2744;
}

.saved-badge {
    background: #00f5d420;
    border: 1px solid #00f5d4;
    border-radius: 6px;
    padding: 6px 12px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.8rem;
    color: #00f5d4;
}
</style>
""", unsafe_allow_html=True)


# ─── Shared State (thread-safe) ────────────────────────────────────────────────
class TrackerState:
    def __init__(self):
        self.lock = threading.Lock()
        self.counts: dict[str, int] = {}
        self.total_unique: set = set()
        self.fps: float = 0.0
        self.alerts: list[str] = []
        self.saved_frames: list[str] = []
        self.frame_times: deque = deque(maxlen=30)
        self.track_history: dict[int, deque] = defaultdict(lambda: deque(maxlen=40))
        self.conf_threshold: float = 0.5
        self.alert_classes: set = set()
        self.save_trigger: set = set()
        self.draw_trails: bool = True
        self.draw_heatmap: bool = False
        self.heatmap_accumulator: np.ndarray | None = None

state = TrackerState()

os.makedirs("saved_frames", exist_ok=True)


# ─── Model Loading (fixed, no sidebar selection) ──────────────────────────────
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")   # always use nano model

model = load_model()


# ─── Color Palette per class ───────────────────────────────────────────────────
CLASS_COLORS: dict[str, tuple] = {}
_rng = np.random.default_rng(42)

def get_class_color(cls_name: str) -> tuple:
    if cls_name not in CLASS_COLORS:
        h = int(_rng.integers(0, 180))
        rgb = cv2.cvtColor(np.uint8([[[h, 220, 200]]]), cv2.COLOR_HSV2BGR)[0][0]
        CLASS_COLORS[cls_name] = (int(rgb[0]), int(rgb[1]), int(rgb[2]))
    return CLASS_COLORS[cls_name]


# ─── Draw Overlays (unchanged) ────────────────────────────────────────────────
def draw_advanced(frame: np.ndarray, results, track_history: dict, draw_trails: bool) -> np.ndarray:
    h, w = frame.shape[:2]
    overlay = frame.copy()

    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return frame

    for box in boxes:
        if box.id is None:
            continue

        track_id = int(box.id[0])
        cls_id = int(box.cls[0])
        cls_name = results[0].names[cls_id]
        conf = float(box.conf[0])
        color = get_class_color(cls_name)

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        # Trail
        track_history[track_id].append((cx, cy))
        if draw_trails and len(track_history[track_id]) > 2:
            pts = list(track_history[track_id])
            for i in range(1, len(pts)):
                alpha = i / len(pts)
                thickness = max(1, int(alpha * 3))
                fade_color = tuple(int(c * alpha) for c in color)
                cv2.line(overlay, pts[i-1], pts[i], fade_color, thickness)

        # Semi-transparent filled box
        sub = overlay[y1:y2, x1:x2]
        colored_rect = np.full_like(sub, color, dtype=np.uint8)
        cv2.addWeighted(colored_rect, 0.12, sub, 0.88, 0, sub)

        # Corner brackets instead of full box
        br_len = min(20, (x2 - x1) // 4, (y2 - y1) // 4)
        thickness = 2
        corners = [
            [(x1, y1), (x1 + br_len, y1), (x1, y1 + br_len)],
            [(x2, y1), (x2 - br_len, y1), (x2, y1 + br_len)],
            [(x1, y2), (x1 + br_len, y2), (x1, y2 - br_len)],
            [(x2, y2), (x2 - br_len, y2), (x2, y2 - br_len)],
        ]
        for corner in corners:
            cv2.line(overlay, corner[0], corner[1], color, thickness)
            cv2.line(overlay, corner[0], corner[2], color, thickness)

        # Label background
        label = f"{cls_name} #{track_id}  {conf:.0%}"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
        pad = 4
        lx, ly = x1, max(y1 - lh - pad * 2, 0)
        cv2.rectangle(overlay, (lx, ly), (lx + lw + pad * 2, ly + lh + pad * 2), color, -1)
        cv2.putText(overlay, label, (lx + pad, ly + lh + pad),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (10, 10, 10), 1, cv2.LINE_AA)

        # Confidence ring around center dot
        radius = max(4, int(conf * 12))
        cv2.circle(overlay, (cx, cy), radius, color, -1)
        cv2.circle(overlay, (cx, cy), radius + 2, (255, 255, 255), 1)

    # Scanline effect (subtle)
    for y in range(0, h, 4):
        cv2.line(overlay, (0, y), (w, y), (0, 0, 0), 1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, overlay)

    return overlay


# ─── Video Callback (unchanged) ────────────────────────────────────────────────
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")

    with state.lock:
        conf = state.conf_threshold
        alert_cls = state.alert_classes
        draw_trails = state.draw_trails
        draw_heatmap = state.draw_heatmap

    t0 = time.perf_counter()

    results = model.track(
        img,
        persist=True,
        conf=conf,
        iou=0.45,
        verbose=False,
        tracker="bytetrack.yaml",
    )

    elapsed = time.perf_counter() - t0

    # Update state
    current_counts: dict[str, int] = defaultdict(int)
    new_alerts: list[str] = []

    boxes = results[0].boxes
    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            cls_id = int(box.cls[0])
            cls_name = results[0].names[cls_id]
            current_counts[cls_name] += 1

            if cls_name in alert_cls:
                tid = int(box.id[0]) if box.id is not None else "?"
                new_alerts.append(f"⚠ {cls_name.upper()} detected (ID #{tid})")

        # Heatmap
        if draw_heatmap:
            h, w = img.shape[:2]
            hmap = np.zeros((h, w), dtype=np.float32)
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = (x1+x2)//2, (y1+y2)//2
                cv2.circle(hmap, (cx, cy), 40, 1.0, -1)
            with state.lock:
                if state.heatmap_accumulator is None or state.heatmap_accumulator.shape != (h, w):
                    state.heatmap_accumulator = hmap
                else:
                    state.heatmap_accumulator = state.heatmap_accumulator * 0.97 + hmap * 0.03

    with state.lock:
        state.counts = dict(current_counts)
        state.frame_times.append(elapsed)
        if len(state.frame_times) > 1:
            state.fps = 1.0 / (sum(state.frame_times) / len(state.frame_times))
        state.alerts = new_alerts[-5:]  # keep last 5

        # Auto-save
        for cls_name in state.save_trigger:
            if cls_name in current_counts:
                ts = datetime.now().strftime("%H%M%S_%f")[:12]
                fname = f"saved_frames/{cls_name}_{ts}.jpg"
                cv2.imwrite(fname, img)
                state.saved_frames.append(fname)
                state.saved_frames = state.saved_frames[-20:]

        th = state.track_history

    # Draw
    annotated = draw_advanced(img, results, th, draw_trails)

    # Overlay heatmap
    if draw_heatmap:
        with state.lock:
            hmap_acc = state.heatmap_accumulator
        if hmap_acc is not None:
            norm = cv2.normalize(hmap_acc, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            colored = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
            annotated = cv2.addWeighted(annotated, 0.7, colored, 0.3, 0)

    # FPS stamp
    with state.lock:
        fps_val = state.fps
    cv2.putText(annotated, f"FPS: {fps_val:.1f}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 245, 212), 2, cv2.LINE_AA)

    return av.VideoFrame.from_ndarray(annotated, format="bgr24")


# ─── Sidebar (model selector removed) ─────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ SETTINGS")
    st.markdown("---")

    # ❌ model_choice selectbox and loading line are GONE

    conf_val = st.slider("Confidence Threshold", 0.1, 0.95, 0.45, 0.05)
    with state.lock:
        state.conf_threshold = conf_val

    st.markdown("---")
    st.markdown("### 🎨 Visualization")

    trails = st.toggle("Motion Trails", value=True)
    heatmap = st.toggle("Heatmap Overlay", value=False)
    with state.lock:
        state.draw_trails = trails
        state.draw_heatmap = heatmap

    st.markdown("---")
    st.markdown("### 🔔 Alert Classes")

    all_coco = [
        "person","bicycle","car","motorcycle","airplane","bus","train","truck",
        "boat","traffic light","fire hydrant","stop sign","parking meter","bench",
        "bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe",
        "backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard",
        "sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
        "tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl",
        "banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza",
        "donut","cake","chair","couch","potted plant","bed","dining table","toilet",
        "tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven",
        "toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear",
        "hair drier","toothbrush" 
    ]

    alert_sel = st.multiselect("Trigger alert when detected:", all_coco, default=["person"])
    with state.lock:
        state.alert_classes = set(alert_sel)

    st.markdown("---")
    st.markdown("### 💾 Auto-Save Frames")
    save_sel = st.multiselect("Save frames when detected:", all_coco, default=[])
    with state.lock:
        state.save_trigger = set(save_sel)

    if st.button("🗑 Clear Saved Frames"):
        with state.lock:
            state.saved_frames = []

# ─── Main Layout (unchanged) ───────────────────────────────────────────────────
st.markdown("""
<h1 style='margin-bottom:0'>🎯 AI OBJECT TRACKER <span style='color:#0090ff'>PRO</span></h1>
<p style='color:#7a8aaa; font-family: Share Tech Mono, monospace; font-size:0.85rem; margin-top:4px'>
YOLOv8 · ByteTrack · Real-Time · Multi-Class
</p>
""", unsafe_allow_html=True)

# Metrics row
m1, m2, m3, m4 = st.columns(4)

def metric_card(col, label, key):
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" id="{key}">—</div>
            <div class="metric-label">{label}</div>
        </div>""", unsafe_allow_html=True)

with m1:
    fps_placeholder = st.empty()
with m2:
    total_placeholder = st.empty()
with m3:
    classes_placeholder = st.empty()
with m4:
    saved_placeholder = st.empty()

st.markdown("<br>", unsafe_allow_html=True)

# Video + info columns
col_vid, col_info = st.columns([3, 1])

with col_vid:
    webrtc_streamer(
        key="adv-tracker",
        mode=WebRtcMode.SENDRECV,
        video_frame_callback=video_frame_callback,
        async_processing=True,
        rtc_configuration={
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]},
                {"urls": ["stun:stun1.l.google.com:19302"]},
            ]
        },
        media_stream_constraints={
            "video": {
                "facingMode": "environment",          # ✅ BACK CAMERA ADDED
                "width": {"ideal": 1280},
                "height": {"ideal": 720},
                "frameRate": {"ideal": 30, "max": 30},
            },
            "audio": False,
        },
    )

with col_info:
    st.markdown("### 📊 Live Counts")
    counts_placeholder = st.empty()
    st.markdown("### 🚨 Alerts")
    alerts_placeholder = st.empty()
    st.markdown("### 💾 Saved")
    saved_list_placeholder = st.empty()

# ─── Auto-refresh dashboard (original while loop – UNCHANGED) ─────────────────
import time as _time

while True:
    with state.lock:
        counts = dict(state.counts)
        fps = state.fps
        alerts = list(state.alerts)
        saved = list(state.saved_frames)

    total_objs = sum(counts.values())
    n_classes = len(counts)
    n_saved = len(saved)

    # Metric cards
    fps_placeholder.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{fps:.1f}</div>
        <div class="metric-label">FPS</div>
    </div>""", unsafe_allow_html=True)

    total_placeholder.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{total_objs}</div>
        <div class="metric-label">Objects</div>
    </div>""", unsafe_allow_html=True)

    classes_placeholder.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{n_classes}</div>
        <div class="metric-label">Classes</div>
    </div>""", unsafe_allow_html=True)

    saved_placeholder.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{n_saved}</div>
        <div class="metric-label">Saved</div>
    </div>""", unsafe_allow_html=True)

    # Object counts
    if counts:
        pills = "".join(
            f'<span class="object-pill">{cls} <b>{cnt}</b></span>'
            for cls, cnt in sorted(counts.items(), key=lambda x: -x[1])
        )
        counts_placeholder.markdown(pills, unsafe_allow_html=True)
    else:
        counts_placeholder.markdown("<span style='color:#7a8aaa; font-size:0.85rem'>No objects detected</span>", unsafe_allow_html=True)

    # Alerts
    if alerts:
        alert_html = "".join(f'<div class="alert-box">{a}</div>' for a in alerts)
        alerts_placeholder.markdown(alert_html, unsafe_allow_html=True)
    else:
        alerts_placeholder.markdown("<span style='color:#7a8aaa; font-size:0.85rem'>All clear</span>", unsafe_allow_html=True)

    # Saved frames list
    if saved:
        saved_html = "".join(
            f'<div style="font-family:Share Tech Mono,monospace;font-size:0.7rem;color:#00f5d4;margin:2px 0">📸 {os.path.basename(f)}</div>'
            for f in reversed(saved[-5:])
        )
        saved_list_placeholder.markdown(saved_html, unsafe_allow_html=True)
    else:
        saved_list_placeholder.markdown("<span style='color:#7a8aaa; font-size:0.75rem'>None yet</span>", unsafe_allow_html=True)

    _time.sleep(0.5)

