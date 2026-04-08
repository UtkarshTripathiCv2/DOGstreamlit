import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import tempfile
import os
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
from fpdf import FPDF
from datetime import datetime
from urllib.parse import quote_plus

# --- Page Configuration ---
st.set_page_config(
    page_title="D.O.G. Vision System | AI Monitoring",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Path Configuration for Streamlit Cloud ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "DOG.pt")

# --- Premium Styling (CSS) ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@600;700&family=Lato:wght@400;700&display=swap');
.stApp {
    background-image: url("https://images.unsplash.com/photo-1495534027489-3543734d35e1?q=80&w=1932&auto-format&fit=crop");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
}
#MainMenu, footer { visibility: hidden; }
h1, h2, h3 { font-family: 'Montserrat', sans-serif; color: #FFFFFF; text-shadow: 2px 2px 6px rgba(0,0,0,0.5); }
p, .stMarkdown { font-family: 'Lato', sans-serif; color: #E0E0E0; }
.main-title { font-size: 3.8rem; font-weight: 700; text-align: center; margin-bottom: 0; }
.sub-header { font-size: 1.5rem; font-weight: 400; text-align: center; color: #FFCA28; margin-bottom: 40px; }
.glass-container { background: rgba(10, 25, 10, 0.7); backdrop-filter: blur(12px); border-radius: 15px; border: 1px solid rgba(255, 255, 255, 0.18); padding: 2rem; }
[data-testid="stSidebar"] { background: rgba(20, 20, 20, 0.8); backdrop-filter: blur(10px); }
.stButton>button { background-color: #FFB300; color: #111111 !important; border-radius: 10px; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

# --- PDF Report Generation ---
class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'D.O.G. Vision System - Analysis Report', 0, 1, 'C')
        self.ln(10)
    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def generate_pdf_report(original_img, annotated_img, detections, uploaded_filename):
    pdf = PDF()
    pdf.add_page()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as orig_f, \
         tempfile.NamedTemporaryFile(delete=False, suffix=".png") as anno_f:
        cv2.imwrite(orig_f.name, original_img)
        cv2.imwrite(anno_f.name, annotated_img)
        pdf.image(orig_f.name, x=10, y=30, w=90)
        pdf.image(anno_f.name, x=110, y=30, w=90)
    
    pdf.set_y(100)
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, f'Analysis for: {uploaded_filename}', 0, 1)
    if detections:
        for item, conf in detections:
            pdf.set_font('Arial', '', 10)
            pdf.cell(0, 10, f'- {item}: {conf:.2f}', 0, 1)
    
    return pdf.output(dest='S').encode('latin-1')

# --- Model Loading ---
@st.cache_resource
def load_model(path):
    if not os.path.exists(path):
        return None
    return YOLO(path)

def annotate_frame(frame, model, confidence_threshold):
    results = model(frame, verbose=False)
    annotated_frame = frame.copy()
    detections_list = []
    detection_color = (0, 191, 255) 

    for r in results:
        for box in r.boxes:
            confidence = box.conf[0]
            if confidence > confidence_threshold:
                cls_id = int(box.cls[0])
                class_name = model.names[cls_id]
                detections_list.append((class_name, float(confidence)))
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), detection_color, 2)
                cv2.putText(annotated_frame, f"{class_name} {confidence:.2f}", (x1, y1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, detection_color, 2)
    return annotated_frame, detections_list

# --- Main App ---
def main():
    st.markdown('<p class="main-title">D.O.G. Vision System</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Agricultural Monitoring</p>', unsafe_allow_html=True)

    model = load_model(MODEL_PATH)
    if model is None:
        st.error(f"Model file not found at {MODEL_PATH}. Please ensure DOG.pt is in the root directory.")
        return

    st.sidebar.title("🌿 Settings")
    conf_level = st.sidebar.slider("Confidence", 0.0, 1.0, 0.5, 0.05)
    source = st.sidebar.radio("Source", ["Image", "Video", "Live"])

    st.markdown('<div class="glass-container">', unsafe_allow_html=True)
    
    if source == "Image":
        file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
        if file:
            img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), 1)
            anno, det = annotate_frame(img, model, conf_level)
            st.image(anno, channels="BGR", use_container_width=True)
            if det:
                pdf_data = generate_pdf_report(img, anno, det, file.name)
                st.download_button("Download Report", pdf_data, "Report.pdf", "application/pdf")

    elif source == "Live":
        st.info("Starting WebRTC Stream...")
        webrtc_streamer(key="live", video_processor_factory=lambda: None) # Simple placeholder

    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
