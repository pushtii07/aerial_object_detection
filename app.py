import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO

# ---------------- PAGE SETUP ----------------
st.set_page_config(page_title="Aerial Detection System", layout="centered")

st.title("🛰 Aerial Object Detection System")
st.write("Upload an image and choose a model to run inference.")

# ---------------- MODEL SELECTION (CENTERED UI) ----------------
model_choice = st.selectbox(
    "Choose Model",
    ["Object Detection (YOLO)",
    "CNN Classification (CNN)"]
)

# ---------------- LOAD YOLO MODEL ----------------
@st.cache_resource
def load_model():
    return YOLO("best.pt")

yolo_model = load_model()

# ---------------- IMAGE UPLOAD ----------------
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    # Show smaller input image
    st.image(image, caption="Uploaded Image", width=300)

    img = np.array(image)

    # ---------------- OBJECT DETECTION ----------------
    if model_choice == "Object Detection (YOLO)":

        results = yolo_model(img)

        result_img = results[0].plot()

        st.subheader("Detection Result")

        # Resize output for clean UI
        st.image(result_img, width=400)
