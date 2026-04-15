import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import tempfile
import os
from streamlit_webrtc import webrtc_streamer
from fpdf import FPDF
from datetime import datetime

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="D.O.G Vision System",
    page_icon="🌿",
    layout="wide"
)

# ---------------- PATH ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "DOG.pt")

# ---------------- CUSTOM UI ----------------
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
}
.main-title {
    font-size: 4rem;
    font-weight: bold;
    text-align: center;
    color: #00e676;
}
.team {
    text-align:center;
    font-size:1.2rem;
    color:#ffffff;
}
.card {
    background: rgba(255,255,255,0.05);
    padding:20px;
    border-radius:15px;
    box-shadow:0 8px 32px rgba(0,0,0,0.3);
}
</style>
""", unsafe_allow_html=True)

# ---------------- PDF ----------------
class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 14)
        self.cell(0, 10, 'D.O.G Vision System Report', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')


def generate_pdf(original, annotated, detections, filename):
    pdf = PDF()
    pdf.add_page()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f1, \
         tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f2:
        cv2.imwrite(f1.name, original)
        cv2.imwrite(f2.name, annotated)
        pdf.image(f1.name, x=10, y=30, w=90)
        pdf.image(f2.name, x=110, y=30, w=90)

    pdf.set_y(110)
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, f'File: {filename}', 0, 1)

    pdf.set_font('Arial', '', 11)
    for d, c in detections:
        pdf.cell(0, 8, f'{d} : {c:.2f}', 0, 1)

    pdf.ln(10)
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Developed By:', 0, 1)

    pdf.set_font('Arial', '', 11)
    pdf.cell(0, 8, 'Utkarsh Tripathi', 0, 1)
    pdf.cell(0, 8, 'Aditya Kumar Raj', 0, 1)
    pdf.cell(0, 8, 'Abhiyanshu Kumar', 0, 1)

    pdf.cell(0, 8, f'Date: {datetime.now()}', 0, 1)

    return pdf.output(dest='S').encode('latin-1')

# ---------------- MODEL ----------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return YOLO(MODEL_PATH)


def detect(frame, model, conf):
    res = model(frame)
    out = frame.copy()
    dets = []

    for r in res:
        for box in r.boxes:
            c = float(box.conf[0])
            if c > conf:
                cls = int(box.cls[0])
                name = model.names[cls]
                dets.append((name, c))
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(out, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(out, f"{name} {c:.2f}", (x1, y1-5), 0, 0.6, (0,255,0), 2)

    return out, dets

# ---------------- MAIN ----------------
def main():
    st.markdown('<div class="main-title">D.O.G Vision System</div>', unsafe_allow_html=True)
    st.markdown('<div class="team">Utkarsh Tripathi | Aditya Kumar Raj | Abhiyanshu Kumar</div>', unsafe_allow_html=True)

    model = load_model()
    if model is None:
        st.error("Model not found")
        return

    st.sidebar.title("Settings")
    conf = st.sidebar.slider("Confidence", 0.0, 1.0, 0.5)
    mode = st.sidebar.selectbox("Mode", ["Image", "Video", "Live"])

    st.markdown('<div class="card">', unsafe_allow_html=True)

    if mode == "Image":
        file = st.file_uploader("Upload Image", type=["jpg","png"])
        if file:
            img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), 1)
            out, det = detect(img, model, conf)

            col1, col2 = st.columns(2)
            col1.image(img, caption="Original")
            col2.image(out, caption="Detected")

            if det:
                st.success(f"Detected {len(det)} objects")
                pdf = generate_pdf(img, out, det, file.name)
                st.download_button("Download Report", pdf, "report.pdf")

    elif mode == "Video":
        file = st.file_uploader("Upload Video", type=["mp4"])
        if file:
            st.info("Processing video...")

    elif mode == "Live":
        st.info("Live detection started")
        webrtc_streamer(key="live")

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Features Added")
    st.write("- Modern UI\n- Team credits\n- PDF with names\n- Side-by-side comparison\n- Detection counter")


if __name__ == "__main__":
    main()
