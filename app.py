import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

st.title("Crop Disease Detection Platform")
st.write("Upload a leaf photo to identify diseases using ResNet50.")

@st.cache_resource
def load_model():
    # This approach is more compatible with newer TensorFlow versions
    with open('config.json', 'r') as f:
        json_config = f.read()

    # Adding a try-except block to handle version mismatches
    try:
        model = tf.keras.models.model_from_json(json_config)
    except Exception:
        # Fallback for newer Keras versions
        import json
        config_dict = json.loads(json_config)
        model = tf.keras.Sequential.from_config(config_dict)

    model.load_weights('model.weights.h5')
    return model
