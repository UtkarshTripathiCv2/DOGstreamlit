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
st.set_page_config(page_title="D.O.G Vision System", page_icon="🌿", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "DOG.pt")

# ---------------- UI STYLE ----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(rgba(0,0,0,0.7), rgba(0,0,0,0.9)),
    url("https://images.unsplash.com/photo-1501004318641-b39e6451bec6");
    background-size: cover;
}
.main-title {
    text-align:center;
    font-size:4rem;
    font-weight:bold;
    background: linear-gradient(90deg,#00e676,#00c853,#69f0ae);
    -webkit-background-clip: text;
    color: transparent;
}
.glass {
    background: rgba(255,255,255,0.05);
    border-radius:20px;
    backdrop-filter: blur(20px);
    padding:25px;
    margin-top:20px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- PDF ----------------
class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 14)
        self.cell(0, 10, 'D.O.G Vision System Report', 0, 1, 'C')

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

# --- Suggestions + Cure Logic ---
def get_suggestion_and_cure(label):
    data = {
        "disease": ("Apply fungicide spray", "Use Carbendazim or Mancozeb weekly"),
        "pest": ("Spray pesticide", "Use Imidacloprid or Neem oil treatment"),
        "dry": ("Increase watering frequency", "Install drip irrigation system"),
        "healthy": ("No action needed", "Maintain current care routine")
    }
    return data.get(label.lower(), ("Monitor plant regularly", "General organic care recommended"))


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

    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Detection Analysis:', 0, 1)

    pdf.set_font('Arial', '', 11)
    for d, c in detections:
        suggestion, cure = get_suggestion_and_cure(d)
        pdf.cell(0, 8, f'{d} ({c:.2f})', 0, 1)
        pdf.cell(0, 8, f'Suggestion: {suggestion}', 0, 1)
        pdf.cell(0, 8, f'Cure/Treatment: {cure}', 0, 1)
        pdf.ln(2)

    pdf.ln(5)
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Final Recommendation:', 0, 1)
    pdf.set_font('Arial', '', 11)
    pdf.multi_cell(0, 8, 'Ensure regular monitoring of crops. Apply suggested treatments promptly to avoid spread of disease. Maintain proper irrigation, sunlight, and nutrient balance for optimal growth.')

    pdf.ln(5)
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
    st.markdown('<p style="text-align:center;color:white;">Utkarsh Tripathi | Aditya Kumar Raj | Abhiyanshu Kumar</p>', unsafe_allow_html=True)

    model = load_model()
    if model is None:
        st.error("Model not found")
        return

    st.sidebar.title("Settings")
    conf = st.sidebar.slider("Confidence", 0.0, 1.0, 0.5)
    mode = st.sidebar.selectbox("Mode", ["Image", "Live"])

    st.markdown('<div class="glass">', unsafe_allow_html=True)

    if mode == "Image":
        st.markdown("### 📤 Upload Image for AI Analysis")
        file = st.file_uploader("", type=["jpg","png","jpeg"])

        if file:
            img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), 1)
            out, det = detect(img, model, conf)

            col1, col2 = st.columns(2)
            col1.image(img, caption="Original")
            col2.image(out, caption="Detected")

            col1, col2, col3 = st.columns(3)
            col1.metric("Objects", len(det))
            col2.metric("Avg Confidence", f"{np.mean([c for _,c in det]):.2f}" if det else "0")
            col3.metric("Status", "Healthy" if len(det)==0 else "Issue Found")

            if det:
                st.subheader("🌱 Suggestions & Cure")
                for d, c in det:
                    suggestion, cure = get_suggestion_and_cure(d)
                    st.write(f"**{d}** → Suggestion: {suggestion}")
                    st.write(f"Cure: {cure}")
                    st.write("---")

                pdf = generate_pdf(img, out, det, file.name)
                st.download_button("📄 Download Full Report", pdf, "report.pdf")

    elif mode == "Live":
        st.info("Live detection started")
        webrtc_streamer(key="live")

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<hr><p style='text-align:center;color:gray;'>Developed by Utkarsh Tripathi • Aditya Kumar Raj • Abhiyanshu Kumar</p>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
