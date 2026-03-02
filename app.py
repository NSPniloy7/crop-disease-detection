import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Page Configuration for better aesthetics
st.set_page_config(page_title="NeuralOfNSP - Crop AI", page_icon="🌿", layout="centered")

# Custom CSS for a clean look
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stAlert {
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🌿 NeuralOfNSP: Crop Disease Detection")

# Experimental Stage Note
st.warning("⚠️ **Disclaimer:** This project is currently in the **Experimental Stage**. Predictions are for educational purposes and should be verified by an agricultural expert.")

@st.cache_resource
def load_model():
    with open('config.json', 'r') as f:
        json_config = f.read()
    try:
        model = tf.keras.models.model_from_json(json_config)
    except Exception:
        import json
        model = tf.keras.Sequential.from_config(json.loads(json_config))
    model.load_weights('model.weights.h5')
    return model

model = load_model()

# This automatically allows "again and again" use as Streamlit reruns 
# the script whenever a new file is dropped into the uploader.
uploaded_file = st.file_uploader("Upload a leaf photo for instant analysis", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Use columns to make it look nicer
    col1, col2 = st.columns(2)
    
    with col1:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Specimen', use_container_width=True)
    
    with col2:
        st.info("🔄 **Processing Image...**")
        # Pre-process
        img = image.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Prediction
        predictions = model.predict(img_array)
        result = np.argmax(predictions)
        confidence = np.max(predictions) * 100
        
        st.success(f"**Analysis Complete!**")
        st.metric(label="Detected Category Index", value=result)
        st.write(f"Confidence Level: {confidence:.2f}%")

st.divider()
st.caption("Developed by Niloy Saha | B.Tech CSE, KIIT University")
