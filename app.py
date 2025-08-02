import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load your trained model
model = load_model('aslmodel.h5')

# Class labels for ASL letters
class_names = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

st.title("ASL Alphabet Letter Predictor")

uploaded_file = st.file_uploader("Upload an ASL hand sign image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Display uploaded image
    img = Image.open(uploaded_file).convert('L')  # convert to grayscale
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Preprocess image for model
    img = img.resize((64, 64))
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 64, 64, 1)  # model expects 4D input

    # Predict
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    st.success(f"Predicted ASL Letter: **{predicted_class}**")
