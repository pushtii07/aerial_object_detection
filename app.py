import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from ultralytics import YOLO

# Load models
model = load_model("mobilenetv2_aerial_model.h5")
model.save("mobilenetv2_fixed.keras")
yolo_model = YOLO("best.pt")



st.title("Aerial Object Classification and Detection")
st.write("Upload an aerial image to classify or detect objects.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

task = st.radio(
    "Select Task",
    ["Bird vs Drone Classification", "Object Detection"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img = np.array(image)

    if task == "Bird vs Drone Classification":

        img_resized = cv2.resize(img, (224,224))
        img_resized = img_resized / 255.0
        img_resized = np.expand_dims(img_resized, axis=0)

        prediction = cnn_model.predict(img_resized)[0][0]

        if prediction > 0.5:
            st.success("Prediction: Drone 🚁")
        else:
            st.success("Prediction: Bird 🐦")

    if task == "Object Detection":

        results = yolo_model(img)
        result_img = results[0].plot()

        st.image(result_img, caption="Detection Result", use_container_width=True)
