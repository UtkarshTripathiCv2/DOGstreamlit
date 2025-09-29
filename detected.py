import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import tempfile
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

# --- Premium Styling (CSS) ---
st.markdown("""
<style>
/* Import Google Fonts for a premium feel */
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@600;700&family=Lato:wght@400;700&display=swap');

/* Main App Styling */
.stApp {
    background-image: url("https://images.unsplash.com/photo-1495534027489-3543734d35e1?q=80&w=1932&auto-format&fit=crop");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
}

/* Hide Streamlit Header/Footer */
.stApp > header {
    background-color: transparent;
}
#MainMenu, footer {
    visibility: hidden;
}

/* --- Typography --- */
h1, h2, h3 {
    font-family: 'Montserrat', sans-serif;
    color: #FFFFFF;
    text-shadow: 2px 2px 6px rgba(0,0,0,0.5);
}

p, .stMarkdown, .st-emotion-cache-1g8sfsg {
    font-family: 'Lato', sans-serif;
    color: #E0E0E0;
}

/* --- Main Title & Header --- */
.main-title {
    font-size: 3.8rem;
    font-weight: 700;
    text-align: center;
    margin-bottom: 0;
}

.sub-header {
    font-size: 1.5rem;
    font-weight: 400;
    text-align: center;
    color: #FFCA28; /* Light Amber for subtitle */
    margin-bottom: 40px;
}

/* --- Glassmorphism Containers --- */
.glass-container {
    background: rgba(10, 25, 10, 0.7);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border-radius: 15px;
    border: 1px solid rgba(255, 255, 255, 0.18);
    padding: 2rem;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
}

/* --- Sidebar Styling --- */
[data-testid="stSidebar"] {
    background: rgba(20, 20, 20, 0.8); /* Darker sidebar */
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border-right: 1px solid rgba(255, 255, 255, 0.1);
}
[data-testid="stSidebar"] h2 {
    color: #FFC107; /* Amber */
}

/* --- Custom Button Styling --- */
.stButton>button, .stLinkButton>a {
    font-family: 'Lato', sans-serif;
    background-color: #FFB300; /* Amber */
    color: #111111 !important; /* Dark Text */
    border-radius: 10px;
    border: 1px solid #FFA000; /* Darker Amber */
    padding: 10px 24px;
    font-weight: 700;
    transition: all 0.3s ease-in-out;
    text-decoration: none;
}
.stButton>button:hover, .stLinkButton>a:hover {
    background-color: #FFA000;
    border-color: #FFCA28;
    transform: translateY(-2px);
    color: #111111 !important;
}
</style>
""", unsafe_allow_html=True)


# --- PDF Report Generation ---
class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'D.O.G. Vision System - Analysis Report', 0, 1, 'C')
        self.set_font('Arial', '', 10)
        self.cell(0, 10, f'Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 0, 1, 'C')
        self.ln(10)

    def footer(self):
        self.set_y(-20)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, 'Report by D.O.G. Vision | Developed by Utkarsh Tripathi, Aditya Kumar Raj & Abhiyanshu Kumar', 0, 0, 'C')
        self.set_y(-15)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def generate_pdf_report(original_img, annotated_img, detections, uploaded_filename):
    pdf = PDF()
    pdf.add_page()
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as orig_f, \
         tempfile.NamedTemporaryFile(delete=False, suffix=".png") as anno_f:
        
        cv2.imwrite(orig_f.name, original_img)
        cv2.imwrite(anno_f.name, annotated_img)

        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, f'Analysis for: {uploaded_filename}', 0, 1)
        
        page_width = pdf.w - 2 * pdf.l_margin
        col_width = page_width / 2 - 5
        
        aspect_ratio = original_img.shape[0] / original_img.shape[1]
        img_height = col_width * aspect_ratio

        pdf.image(orig_f.name, x=pdf.l_margin, y=pdf.get_y(), w=col_width)
        pdf.image(anno_f.name, x=pdf.l_margin + col_width + 10, y=pdf.get_y(), w=col_width)
        
        pdf.set_y(pdf.get_y() + img_height + 10)
        
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Detection Summary', 0, 1)
    
    if detections:
        pdf.set_font('Arial', 'B', 10)
        pdf.cell(60, 10, 'Detected Object', 1, 0, 'C')
        pdf.cell(40, 10, 'Confidence', 1, 0, 'C')
        pdf.cell(90, 10, 'Web Search for Treatment', 1, 1, 'C')

        pdf.set_font('Arial', '', 10)
        for item, conf in sorted(detections, key=lambda x: x[1], reverse=True):
            search_query = quote_plus(f"how to cure {item}")
            search_url = f"https://www.google.com/search?q={search_query}"
            pdf.cell(60, 10, item, 1, 0)
            pdf.cell(40, 10, f'{conf:.2f}', 1, 0, 'C')
            pdf.set_text_color(0, 0, 255)
            pdf.set_font('Arial', 'U', 10)
            pdf.cell(90, 10, "Click Here for Treatment Info", 1, 1, 'C', link=search_url)
            pdf.set_text_color(0, 0, 0)
            pdf.set_font('Arial', '', 10)

    else:
        pdf.set_font('Arial', '', 10)
        pdf.cell(0, 10, 'No objects detected above the confidence threshold.', 0, 1)
        
    # Robustly handle FPDF output for Streamlit deployment
    output = pdf.output(dest='S')
    if isinstance(output, str):
        return output.encode('latin1')
    return output


# --- Caching and Model Loading ---
@st.cache_resource
def load_model(model_path):
    """Loads and caches the YOLOv8 model."""
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# --- Annotation Logic ---
def annotate_frame(frame, model, confidence_threshold):
    """Annotates a frame with refined YOLOv8 detections using a single accent color."""
    results = model(frame, verbose=False)
    annotated_frame = frame.copy()
    detections_list = []
    
    detection_color = (0, 191, 255) # BGR for Amber/Gold

    for r in results:
        for box in r.boxes:
            confidence = box.conf[0]
            if confidence > confidence_threshold:
                cls_id = int(box.cls[0])
                class_name = model.names[cls_id]
                detections_list.append((class_name, float(confidence)))

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                l = int(min(x2 - x1, y2 - y1) * 0.2)
                t = 2
                cv2.line(annotated_frame, (x1, y1), (x1 + l, y1), detection_color, t); cv2.line(annotated_frame, (x1, y1), (x1, y1 + l), detection_color, t)
                cv2.line(annotated_frame, (x2, y1), (x2 - l, y1), detection_color, t); cv2.line(annotated_frame, (x2, y1), (x2, y1 + l), detection_color, t)
                cv2.line(annotated_frame, (x1, y2), (x1 + l, y2), detection_color, t); cv2.line(annotated_frame, (x1, y2), (x1, y2 - l), detection_color, t)
                cv2.line(annotated_frame, (x2, y2), (x2 - l, y2), detection_color, t); cv2.line(annotated_frame, (x2, y2), (x2, y2 - l), detection_color, t)

                label = f'{class_name} {confidence:.2f}'
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                label_y = y1 - 10 if y1 - 10 > h else y1 + h + 10
                cv2.rectangle(annotated_frame, (x1, label_y - h - 5), (x1 + w, label_y + 5), detection_color, -1)
                cv2.putText(annotated_frame, label, (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2)

    return annotated_frame, detections_list

# --- UI Components ---
def display_image_uploader(model):
    """Component for handling image uploads and displaying results."""
    st.header("Upload an Image for Analysis")
    uploaded_file = st.file_uploader("Choose a file...", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    
    if uploaded_file:
        bytes_data = uploaded_file.getvalue()
        frame = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        
        with st.spinner("🛰️ Analyzing your field..."):
            annotated_frame, detections = annotate_frame(frame, model, st.session_state.confidence)
        
        st.success(f"✅ Analysis complete! Found {len(detections)} objects.")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Image")
            st.image(frame, channels="BGR", use_container_width=True)
        with col2:
            st.subheader("Detection Results")
            st.image(annotated_frame, channels="BGR", use_container_width=True)
        
        st.markdown("---")
        pdf_bytes = generate_pdf_report(frame, annotated_frame, detections, uploaded_file.name)
        st.download_button(
            label="📄 Download Full Report (PDF)",
            data=pdf_bytes,
            file_name=f"DOG_Report_{uploaded_file.name.split('.')[0]}.pdf",
            mime="application/pdf"
        )

        st.subheader("Detected Objects & Actions")
        if detections:
            for item, conf in sorted(detections, key=lambda x: x[1], reverse=True):
                search_query = quote_plus(f"treatment for {item}")
                search_url = f"https://www.google.com/search?q={search_query}"
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"- **{item}** (Confidence: `{conf:.2f}`)")
                with col2:
                    st.link_button("Find Treatment ↗️", search_url)
        else:
            st.info("No objects were detected above the current confidence threshold.")


def display_video_uploader(model):
    """Component for handling video uploads, processing, and displaying results."""
    st.header("Upload a Video for Analysis")
    uploaded_file = st.file_uploader("Choose a video file...", type=["mp4", "mov", "avi", "mkv"], label_visibility="collapsed")
    
    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_file.read())
        
        cap = cv2.VideoCapture(tfile.name)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        st_frame = st.empty()
        progress_bar = st.progress(0)
        
        output_tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_tfile.name, fourcc, fps, (width, height))

        for i in range(frame_count):
            ret, frame = cap.read()
            if not ret: break
            
            annotated_frame, _ = annotate_frame(frame, model, st.session_state.confidence)
            st_frame.image(annotated_frame, channels="BGR", use_container_width=True)
            progress_bar.progress((i + 1) / frame_count)
            out.write(annotated_frame)

        cap.release()
        out.release()
        
        st.success("✅ Video analysis complete!")
        st.subheader("Download Processed Video")
        with open(output_tfile.name, "rb") as f:
            st.download_button("Download as MP4", f, f"detected_{uploaded_file.name}.mp4", "video/mp4")

def display_webcam_feed(model):
    """Component for handling the live webcam feed."""
    st.header("Live Webcam Feed")
    st.info("Click 'START' to activate your camera.")

    class VideoTransformer(VideoTransformerBase):
        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            annotated_frame, _ = annotate_frame(img, model, st.session_state.confidence)
            return annotated_frame
            
    webrtc_streamer(
        key="webcam", 
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=VideoTransformer,
        media_stream_constraints={"video": True, "audio": False},
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

# --- Main Application ---
def main():
    st.markdown('<p class="main-title">D.O.G. Vision System</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Agricultural Monitoring</p>', unsafe_allow_html=True)

    model = load_model("DOG.pt")
    if not model:
        st.error("Model file 'DOG.pt' not found. Please ensure it is in the same directory.")
        return

    st.sidebar.title("🌿 D.O.G. Vision")
    
    if 'confidence' not in st.session_state:
        st.session_state.confidence = 0.5
    
    st.session_state.confidence = st.sidebar.slider(
        "Confidence Threshold", 0.0, 1.0, st.session_state.confidence, 0.05
    )
    
    st.sidebar.markdown("---")
    source_option = st.sidebar.radio(
        "Input Source", 
        ["🖼️ Image Analysis", "📹 Video Analysis", "🛰️ Live Monitoring"],
        index=0
    )
    st.sidebar.markdown("---")
    st.sidebar.info(
        """
        This custom YOLOv8n model was trained on over 50,000 images, 
        achieving an accuracy of approximately 65% for agricultural monitoring.
        """
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        **Developed by:**
        - Utkarsh Tripathi
        - Aditya Kumar Raj
        - Abhiyanshu Kumar
        """
    )
    st.sidebar.markdown("---")
    st.sidebar.warning(
        """
        **Project Vision:** This web application serves as a showcase for our custom-trained dataset. 
        The ultimate goal is to integrate this high-accuracy detection model into our 
        **'Dog On Gears' (D.O.G.)** agricultural robot.
        """
    )

    st.markdown('<div class="glass-container">', unsafe_allow_html=True)
    if source_option == "🖼️ Image Analysis":
        display_image_uploader(model)
    elif source_option == "📹 Video Analysis":
        display_video_uploader(model)
    elif source_option == "🛰️ Live Monitoring":
        display_webcam_feed(model)
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()

