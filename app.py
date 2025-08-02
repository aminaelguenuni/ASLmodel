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
    img = Image.open(uploaded_file)

    # Change this to 'RGB' or 'L' depending on model.input_shape
    img = img.convert('L')  
    st.image(img, caption='Uploaded Image', use_container_width=True)

    img = img.resize((64, 64))
    img_array = np.array(img) / 255.0
    img_array = img_array.astype('float32')
    img_array = img_array.reshape(1, 64, 64, 1)  # or (1,64,64,3) if RGB

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    st.success(f"Predicted ASL Letter: **{predicted_class}**")

