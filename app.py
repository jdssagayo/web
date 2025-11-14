import streamlit as st
import tensorflow as tf
from ultralytics import YOLO
from PIL import Image
import numpy as np
import os
import time
# import requests # No longer needed
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
# The models are in the 'runs' folder, relative to this app.py
MODEL_DIR = "runs" 

# --- (NEW) LFS File Check ---
def check_lfs_files():
    """
    Checks if model files are LFS pointers (small size) instead of full files.
    """
    lfs_warning = """
    🚨 **Git LFS Error:** Your model files seem to be LFS pointers, not the actual models.
    
    This happens when Git LFS isn't set up correctly before pushing to GitHub.
    
    **To Fix This:**
    1.  On your local computer, open your terminal.
    2.  Run `git lfs install`
    3.  Run `git lfs track "*.keras" "*.h5" "*.pt"`
    4.  Run `git add .gitattributes`
    5.  Run `git commit -m "Fix LFS tracking"`
    6.  Run `git push` to re-upload the files.
    7.  Finally, reboot your Streamlit app.
    """
    
    model_paths = [
        os.path.join(MODEL_DIR, 'mobilenetv3_finetuned.keras'),
        os.path.join(MODEL_DIR, 'simple_cnn.h5'),
        os.path.join(MODEL_DIR, 'best.pt')
    ]
    
    for path in model_paths:
        if os.path.exists(path):
            # If file is < 10KB, it's almost certainly a pointer
            if os.path.getsize(path) < 10 * 1024: 
                st.warning(lfs_warning, icon="⚠️")
                return # Show warning once
        else:
            # Show a different warning if file is just missing
            st.error(f"File not found: {path}. Please make sure it's in the '{MODEL_DIR}' folder.", icon="🚨")

# Run the LFS check on app startup
check_lfs_files()

# --- (MODIFIED) Load Models ---
@st.cache_resource
def load_yolo_model():
    # Load from the local path
    local_path = os.path.join(MODEL_DIR, 'best.pt') 
    if not os.path.exists(local_path): return None # Already warned by check_lfs_files
    try:
        model = YOLO(local_path)
        return model
    except Exception as e:
        st.error(f"Error loading YOLO model ({local_path}): {e}")
    return None

@st.cache_resource
def load_mobilenet_model():
    # Load from the local path
    local_path = os.path.join(MODEL_DIR, 'mobilenetv3_finetuned.keras')
    if not os.path.exists(local_path): return None # Already warned by check_lfs_files
    try:
        model = tf.keras.models.load_model(local_path)
        return model
    except Exception as e:
        st.error(f"Error loading MobileNetV3 model ({local_path}): {e}")
    return None

@st.cache_resource
def load_cnn_model():
    # Load from the local path
    local_path = os.path.join(MODEL_DIR, 'simple_cnn.h5')
    if not os.path.exists(local_path): return None # Already warned by check_lfs_files
    try:
        model = tf.keras.models.load_model(local_path)
        return model
    except Exception as e:
        st.error(f"Error loading Simple CNN model ({local_path}): {e}")
    return None

@st.cache_resource
def load_general_detector():
    return YOLO('yolov8n.pt') # This one downloads itself

# --- (MODIFIED) Preprocessing ---
def preprocess_image_for_keras(img_pil, model_name):
    """Prepares a PIL image for MobileNet/CNN prediction."""
    img_rgb = img_pil.convert('RGB')
    img = img_rgb.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    # --- THIS IS THE FIX ---
    # Apply the correct normalization for each model
    if model_name == 'MobileNetV3':
        # MobileNetV3 expects pixels in the range [-1, 1]
        img_array = (img_array / 127.5) - 1.0
    elif model_name == 'Simple':
        # Assuming your simple CNN expects [0, 1]
        img_array = img_array / 255.0
    
    return img_array

# --- Non-Waste Detection Logic ---
def is_likely_non_waste(img_pil, detector):
# ... (rest of this function is unchanged) ...
    """
    Checks if an image contains common non-waste objects (people, animals, etc.).
    Returns (True, "object_name") if non-waste is detected.
    """
    results = detector(img_pil, verbose=False)
    # COCO class IDs for common non-waste items
    non_waste_classes = [ 
        0, # person
        1, 2, 3, 4, 5, 6, 7, 8, # vehicle (bicycle, car, ...)
        15, 16, 17, 18, 19, 20, 21, 22, 23, # animal (bird, cat, dog, ...)
        56, 57, 58, 59, 60, 61, 62, 63 # indoor (chair, couch, ...)
    ]
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            conf = float(box.conf[0])
            if class_id in non_waste_classes and conf > 0.5:
                return True, detector.names[class_id]
    return False, None

# --- Main UI ---
st.title("♻️ Baguio City Waste Classification System")
st.write("Upload an image or use your camera to classify waste.")

with st.sidebar:
# ... (rest of this block is unchanged) ...
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
# ... (rest of this block is unchanged) ...
    uploaded_file = st.file_uploader("Choose a waste image...", type=["jpg", "jpeg", "png", "webp", "jfif"])
with tab2:
# ... (rest of this block is unchanged) ...
    if enable_camera:
        camera_file = st.camera_input("Take a picture")
    else:
        st.info("Live camera is disabled. Enable it in the sidebar settings.")

img_pil = None
if uploaded_file: img_pil = Image.open(uploaded_file)
elif camera_file: img_pil = Image.open(camera_file) 

if img_pil is not None:
    col1, col2 = st.columns(2)
# ... (rest of this block is unchanged) ...
    with col1:
        st.image(img_pil, caption="Input Image", use_column_width=True)
    with col2:
# ... (rest of this block is unchanged) ...
        if st.button("Classify Waste", use_container_width=True, type="primary"):
            is_not_waste = False
            detected_object = None
            
            # 1. Safety Filter Check
            if enable_safety_filter:
# ... (rest of this block is unchanged) ...
                with st.spinner("Checking for non-waste objects..."):
                    detector = load_general_detector()
                    is_not_waste, detected_object = is_likely_non_waste(img_pil, detector)
            
            # 2. Display Safety Warning OR Classify
            if is_not_waste:
# ... (rest of this block is unchanged) ...
                st.error(f"⚠️ **Alert: Non-Waste Detected**")
                st.warning(f"The system detected a **{detected_object}**. This does not appear to be waste.")
                st.info("Please upload an image of waste material (bottles, paper, plastic, etc.).")
            else:
                # 3. Classify Waste
                prediction = "Error"
                confidence = 0.0
                model_name = model_choice.split(' ')[0]
                
                with st.spinner(f"Classifying with {model_name}..."):
# ... (rest of this block is unchanged) ...
                    start_time = time.time()
                    
                    if model_name == 'YOLOv8-Cls':
                        model = load_yolo_model()
                        if model:
# ... (rest of this block is unchanged) ...
                            results = model(img_pil, verbose=False) 
                            probs = results[0].probs
                            confidence = probs.top1conf.item() 
                            prediction_idx = probs.top1
                            prediction = CLASS_NAMES[prediction_idx]
                            
                    elif model_name == 'MobileNetV3':
                        model = load_mobilenet_model()
                        if model:
                            # --- THIS IS THE FIX ---
                            # Pass the model name for correct preprocessing
                            processed_img = preprocess_image_for_keras(img_pil, model_name)
                            probs = model.predict(processed_img, verbose=0)[0]
                            confidence = np.max(probs)
                            prediction_idx = np.argmax(probs)
                            prediction = CLASS_NAMES[prediction_idx]
                            
                    elif model_name == 'Simple':
                        model = load_cnn_model()
                        if model:
                            # --- THIS IS THE FIX ---
                            # Pass the model name for correct preprocessing
                            processed_img = preprocess_image_for_keras(img_pil, model_name)
                            probs = model.predict(processed_img, verbose=0)[0]
                            confidence = np.max(probs)
                            prediction_idx = np.argmax(probs)
                            prediction = CLASS_NAMES[prediction_idx]
                    
                    end_time = time.time()
                    inference_time = (end_time - start_time) * 1000
                
                # 4. Display Results
                st.subheader(f"Prediction: {model_name}")
                if confidence < 0.45: 
# ... (rest of this block is unchanged) ...
                    st.warning(f"⚠️ **Low Confidence ({confidence*100:.2f}%)**")
                    st.write(f"The model thinks this is **{prediction}**, but is not sure.")
                    st.info("Please ensure the waste item is centered and clearly visible.")
                else:
                    # Display formatted results
                    if prediction == "recyclable":
# ... (rest of this block is unchanged) ...
                        st.success(f"**{prediction.upper()}** (Confidence: {confidence*100:.2f}%)")
                        st.info("✅ **Baguio City Guideline:** Place in **Recyclable** bin.\n\n*Examples: Bottles, cans, paper, cardboard.*")
                    elif prediction == "hazardous":
# ... (rest of this block is unchanged) ...
                        st.error(f"**{prediction.upper()}** (Confidence: {confidence*100:.2f}%)")
                        st.warning("☢️ **DANGER:** Do not trash! Take to hazardous waste drop-off.\n\n*Examples: Batteries, electronics, paint.*")
                    elif prediction == "biodegradable":
# ... (rest of this block is unchanged) ...
                        st.success(f"**{prediction.upper()}** (Confidence: {confidence*100:.2f}%)")
                        st.info("✅ **Baguio City Guideline:** Place in **Biodegradable** bin.\n\n*Examples: Food scraps, leaves, paper (soiled).*")
                    else: # non_biodegradable
# ... (rest of this block is unchanged) ...
                        st.info(f"**{prediction.upper()}** (Confidence: {confidence*100:.2f}%)")
                        st.info("ℹ️ **Baguio City Guideline:** Place in **Residual/Landfill** bin.\n\n*Examples: Styrofoam, candy wrappers, diapers.*")
                
                st.caption(f"Inference time: {inference_time:.2f} ms")
