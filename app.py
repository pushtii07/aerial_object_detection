import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from ultralytics import YOLO
import numpy as np
import torchvision.transforms as transforms

# ---------------- PAGE ----------------
st.title("🛰 Aerial Detection: YOLO + CNN Hybrid")

# ---------------- CNN MODEL ----------------
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 56 * 56, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


cnn_model = CNN()
cnn_model.eval()

# ---------------- YOLO MODEL ----------------
# You must have a YOLO weights file OR use pretrained
yolo_model = YOLO("yolov8n.pt")   # lightweight pretrained model

# ---------------- TRANSFORM ----------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

labels = ["Class 0", "Class 1"]

# ---------------- UI ----------------
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Input Image", use_container_width=True)

    # ================= CNN PREDICTION =================
    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        cnn_output = cnn_model(img_tensor)
        cnn_pred = torch.argmax(cnn_output, dim=1).item()

    st.subheader("🧠 CNN Prediction")
    st.success(labels[cnn_pred])

    # ================= YOLO PREDICTION =================
    st.subheader("🎯 YOLO Detection")

    # convert PIL → numpy (NO cv2 used)
    img_np = np.array(image)

    results = yolo_model(img_np)

    # show results image (Ultralytics handles rendering)
    for r in results:
        annotated_img = r.plot()

    st.image(annotated_img, caption="YOLO Detection Result", use_container_width=True)
