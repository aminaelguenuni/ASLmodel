import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

model = load_model('aslmodel.h5')

class_names = sorted([
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
    'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
    'W', 'X', 'Y', 'Z'
]) 

st.title("ASL Alphabet Letter Predictor")

uploaded_file = st.file_uploader("Upload an ASL hand sign image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')  # Convert to RGB for 3 channels
    st.image(img, caption='Uploaded Image', use_container_width=True)

    # Resize and preprocess
    img = img.resize((64, 64))
    img_array = np.array(img) / 255.0  # normalize to [0,1]
    img_array = img_array.astype('float32')
    img_array = np.expand_dims(img_array, axis=0)  # shape (1, 64, 64, 3)

    # Debug info
    st.write("Input shape:", img_array.shape)
    st.write("Model input shape:", model.input_shape)

    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    predicted_class = class_names[predicted_index]

    st.success(f"Predicted ASL Letter: **{predicted_class}**")

