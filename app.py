import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

st.title("Crop Disease Detection")

# Function to load the model
@st.cache_resource
def load_model():
    # Loading model from the json config and weights
    with open('config.json', 'r') as f:
        json_config = f.read()
    model = tf.keras.models.model_from_json(json_config)
    model.load_weights('model.weights.h5')
    return model

model = load_model()

uploaded_file = st.file_uploader("Choose a leaf image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Preprocessing (matching your ResNet50 training)
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Prediction
    predictions = model.predict(img_array)
    st.write(f"Prediction result: {np.argmax(predictions)}")
