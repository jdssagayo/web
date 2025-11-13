import streamlit as st
import tensorflow as tf
from ultralytics import YOLO
from PIL import Image
import numpy as np
import os
import time
import requests # For downloading
import warnings

# --- Configuration ---
st.set_page_config(
    page_title="Baguio City Waste Classifier",
    page_icon="♻️",
    layout="centered"
)
# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')

# --- Class Names ---
CLASS_NAMES = ['biodegradable', 'hazardous', 'non_biodegradable', 'recyclable']
MODEL_DIR = os.path.join(os.getcwd(), 'models_cache') # Cache models here
os.makedirs(MODEL_DIR, exist_ok=True) 

# --- (NEW) Robust Google Drive Download Function ---
@st.cache_data(show_spinner=False) # Cache the download
def download_model(file_id, local_path):
    if not os.path.exists(local_path):
        with st.spinner(f"Downloading {os.path.basename(local_path)}... (This happens once)"):
            URL = "https://docs.google.com/uc?export=download"
            session = requests.Session()
            
            try:
                # First, send a request to get the download confirmation cookie
                response = session.get(URL, params={"id": file_id}, stream=True)
                token = None
                for key, value in response.cookies.items():
                    if key.startswith("download_warning"):
                        token = value
                        break
                
                # If a token was found (meaning Google showed a warning),
                # send a second request with the confirmation token
                if token:
                    params = {"id": file_id, "confirm": token}
                    response = session.get(URL, params=params, stream=True)
                
                # Now, save the file content
                with open(local_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=32768):
                        if chunk:
                            f.write(chunk)
            except Exception as e:
                st.error(f"Error downloading model {os.path.basename(local_path)}: {e}")
                return None
    return local_path

# --- (MODIFIED) Load Models ---
@st.cache_resource
def load_yolo_model():
    local_path = os.path.join(MODEL_DIR, 'best.pt')
    # This is your YOLO model's File ID
    file_id = "1ape52G9_LQqsTMSCNhUXXwX_yfC-EpJR" 
    if download_model(file_id, local_path):
        try:
            return YOLO(local_path)
        except Exception as e:
            st.error(f"Error loading YOLO model ({local_path}): {e}")
    return None

@st.cache_resource
def load_mobilenet_model():
    local_path = os.path.join(MODEL_DIR, 'mobilenetv3_finetuned.keras')
    # This is your NEW MobileNet File ID
    file_id = "1kTvKJ-HufN4IH3KKXfxsYc03BM-hygG1" 
    if download_model(file_id, local_path):
        try:
            return tf.keras.models.load_model(local_path)
        except Exception as e:
            st.error(f"Error loading MobileNetV3 model ({local_path}): {e}")
    return None

@st.cache_resource
def load_cnn_model():
    local_path = os.path.join(MODEL_DIR, 'simple_cnn.h5')
    # This is your NEW CNN File ID
    # !!! WARNING: This is the SAME ID as MobileNet !!!
    file_id = "1kTvKJ-HufN4IH3KKXfxsYc03BM-hygG1" 
    if download_model(file_id, local_path):
        try:
            return tf.keras.models.load_model(local_path)
        except Exception as e:
            st.error(f"Error loading Simple CNN model ({local_path}): {e}")
    return None

@st.cache_resource
def load_general_detector():
    return YOLO('yolov8n.pt') # This one downloads itself

# --- Preprocessing ---
def preprocess_image_for_keras(img_pil):
    img_rgb = img_pil.convert('RGB')
    img = img_rgb.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    # img_array = img_array / 255.0 # Keep this commented out
    return img_array

# --- Non-Waste Detection Logic ---
def is_likely_non_waste(img_pil, detector):
    results = detector(img_pil, verbose=False)
    non_waste_classes = [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 15, 16, 17, 18, 19, 20, 21, 22, 23, 56, 57, 58, 59, 60, 61, 62, 63 ]
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            conf = float(box.conf[0])
            if class_id in non_waste_classes and conf > 0.5:
                return True, detector.names[class_id]
    return False, None

# --- Main UI (No changes below this line) ---
st.title("♻️ Baguio City Waste Classification System")
st.write("Upload an image or use your camera to classify waste.")

with st.sidebar:
    st.header("Settings")
    model_choice = st.radio(
        "Choose Model:",
        ('YOLOv8-Cls (98.1%)', 'Simple CNN (88.3%)', 'MobileNetV3 (78.9%)')
    )
    enable_safety_filter = st.checkbox("Enable Non-Waste Filter", value=True, help="Uses a general object detector to flag people, animals, and vehicles.")
    enable_camera = st.checkbox("Enable Live Camera", value=True, help="Turn the live camera feed on or off.")
    st.divider()
    st.info("This app compares models for the Baguio City waste management thesis.")

tab1, tab2 = st.tabs(["📁 Upload Image", "📸 Live Camera"])
camera_file = None 
with tab1:
    uploaded_file = st.file_uploader("Choose a waste image...", type=["jpg", "jpeg", "png", "webp", "jfif"])
with tab2:
    if enable_camera:
        camera_file = st.camera_input("Take a picture")
    else:
        st.info("Live camera is disabled. Enable it in the sidebar settings.")

img_pil = None
if uploaded_file: img_pil = Image.open(uploaded_file)
elif camera_file: img_pil = Image.open(camera_file) 

if img_pil is not None:
    col1, col2 = st.columns(2)
    with col1:
        st.image(img_pil, caption="Input Image", use_column_width=True)
    with col2:
        if st.button("Classify Waste", use_container_width=True, type="primary"):
            is_not_waste = False
            detected_object = None
            if enable_safety_filter:
                with st.spinner("Checking for non-waste objects..."):
                    detector = load_general_detector()
                    is_not_waste, detected_object = is_likely_non_waste(img_pil, detector)
            if is_not_waste:
                st.error(f"⚠️ **Alert: Non-Waste Detected**")
                st.warning(f"The system detected a **{detected_object}**. This does not appear to be waste.")
                st.info("Please upload an image of waste material (bottles, paper, plastic, etc.).")
            else:
                prediction = "Error"
                confidence = 0.0
                model_name = model_choice.split(' ')[0]
                with st.spinner(f"Classifying with {model_name}..."):
                    start_time = time.time()
                    if model_name == 'YOLOv8-Cls':
                        model = load_yolo_model()
                        if model:
                            results = model(img_pil, verbose=False) 
                            probs = results[0].probs
                            confidence = probs.top1conf.item() 
                            prediction_idx = probs.top1
                            prediction = CLASS_NAMES[prediction_idx]
                    elif model_name == 'MobileNetV3':
                        model = load_mobilenet_model()
                        if model:
                            processed_img = preprocess_image_for_keras(img_pil)
                            probs = model.predict(processed_img, verbose=0)[0]
                            confidence = np.max(probs)
                            prediction_idx = np.argmax(probs)
                            prediction = CLASS_NAMES[prediction_idx]
                    elif model_name == 'Simple':
                        model = load_cnn_model()
                        if model:
                            processed_img = preprocess_image_for_keras(img_pil)
                            probs = model.predict(processed_img, verbose=0)[0]
                            confidence = np.max(probs)
                            prediction_idx = np.argmax(probs)
                            prediction = CLASS_NAMES[prediction_idx]
                    end_time = time.time()
                    inference_time = (end_time - start_time) * 1000
                st.subheader(f"Prediction: {model_name}")
                if confidence < 0.45: 
                    st.warning(f"⚠️ **Low Confidence ({confidence*100:.2f}%)**")
                    st.write(f"The model thinks this is **{prediction}**, but is not sure.")
                    st.info("Please ensure the waste item is centered and clearly visible.")
                else:
                    if prediction == "recyclable":
                        st.success(f"**{prediction.upper()}** (Confidence: {confidence*100:.2f}%)")
                        st.info("✅ **Baguio City Guideline:** Place in **Recyclable** bin.\n\n*Examples: Bottles, cans, paper, cardboard.*")
                    elif prediction == "hazardous":
                        st.error(f"**{prediction.upper()}** (Confidence: {confidence*100:.2f}%)")
                        st.warning("☢️ **DANGER:** Do not trash! Take to hazardous waste drop-off.\n\n*Examples: Batteries, electronics, paint.*")
                    elif prediction == "biodegradable":
                        st.success(f"**{prediction.upper()}** (Confidence: {confidence*100:.2f}%)")
                        st.info("✅ **Baguio City Guideline:** Place in **Biodegradable** bin.\n\n*Examples: Food scraps, leaves, paper (soiled).*")
                    else: # non_biodegradable
                        st.info(f"**{prediction.upper()}** (Confidence: {confidence*100:.2f}%)")
                        st.info("ℹ️ **Baguio City Guideline:** Place in **Residual/Landfill** bin.\n\n*Examples: Styrofoam, candy wrappers, diapers.*")
                st.caption(f"Inference time: {inference_time:.2f} ms")
