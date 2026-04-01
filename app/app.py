import sys
import os
import gdown
import numpy as np
import cv2
import streamlit as st
from PIL import Image

# --- 1. SYSTEM PATH SETUP ---
# This allows the app to find your 'src' folder
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# Now import your custom modules
from src.model import load_trained_model
from src.data_loader import preprocess_image_for_inference
from src.metrics import make_gradcam_heatmap, generate_gradcam_overlay

# --- 2. CLOUD-SAFE PATH CONFIGURATION ---
# Use /tmp to avoid permission errors on Streamlit Cloud
MODEL_DIR = "/tmp/saved_models"
MODEL_PATH = os.path.join(MODEL_DIR, 'advanced_densenet.keras')

# Create the temp folder if it doesn't exist
os.makedirs(MODEL_DIR, exist_ok=True)

# --- 3. MODEL DOWNLOAD & LOAD LOGIC ---
@st.cache_resource
def get_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model from Google Drive (127MB)... This only happens once."):
            file_id = '1phkCm78u090s7Otrjy2rOxgp5S9VpyNd'
            url = f'https://drive.google.com/uc?export=download&id={file_id}'
            try:
                gdown.download(url, MODEL_PATH, quiet=False)
            except Exception as e:
                st.error(f"Download failed: {e}")
                st.stop()
    
    return load_trained_model(MODEL_PATH)

# --- 4. APP UI SETUP ---
st.set_page_config(page_title="Brain Tumor AI", layout="wide")
st.title("🧠 Brain Tumor Diagnostic Assistant with Explainable AI")

st.markdown("""
Upload a Brain MRI scan. The AI will classify the tumor type and generate a **Grad-CAM heatmap** to highlight the specific region of the brain that influenced its decision.
""")

# Load the model
try:
    model = get_model()
    CLASS_NAMES = ['glioma', 'meningioma', 'no_tumor', 'pituitary']
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# --- 5. IMAGE PROCESSING ---
uploaded_file = st.file_uploader("Upload MRI Scan", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Convert uploaded file to OpenCV format
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    st.markdown("### Analysis Results")
    col1, col2, col3 = st.columns(3)
    
    with st.spinner("Analyzing scan..."):
        # Preprocess
        img_array, processed_img = preprocess_image_for_inference(img)
        
        # Inference & Heatmap
        heatmap, predictions = make_gradcam_heatmap(img_array, model)
        
        # Get results
        idx = np.argmax(predictions)
        label = CLASS_NAMES[idx].capitalize()
        conf = predictions[idx] * 100
        
        # Generate Grad-CAM Overlay
        overlay = generate_gradcam_overlay(processed_img, heatmap)

    # Display Results
    with col1:
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Original MRI", use_container_width=True)
    with col2:
        st.image(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB), caption="Preprocessed (Cropped)", use_container_width=True)
    with col3:
        st.image(overlay, caption="Grad-CAM Explainability", use_container_width=True)

    st.success(f"**Diagnosis:** {label} | **Confidence:** {conf:.2f}%")
