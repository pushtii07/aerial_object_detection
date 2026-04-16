import streamlit as st
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO
from tensorflow.keras.models import load_model

st.title("Aerial Object Classification and Detection")

# Load models
cnn_model = load_model("cnn_model.keras", compile=False, safe_mode=False)
yolo_model = YOLO("best.pt")

task = st.radio(
    "Select Task",
    ["Bird vs Drone Classification", "Object Detection"]
)

uploaded_file = st.file_uploader("Upload Image", type=["jpg","jpeg","png"])

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    img = np.array(image)

    st.image(image, caption="Uploaded Image")

    if task == "Bird vs Drone Classification":

        img_resized = cv2.resize(img, (128,128))
        img_resized = img_resized / 255.0
        img_resized = np.expand_dims(img_resized, axis=0)

        prediction = cnn_model.predict(img_resized)

        class_names = ["Bird","Drone"]

        predicted_class = class_names[np.argmax(prediction)]

        st.success(f"Prediction: {predicted_class}")

    elif task == "Object Detection":

        results = yolo_model(img)

        result_img = results[0].plot()

        st.image(result_img, caption="Detection Result")
