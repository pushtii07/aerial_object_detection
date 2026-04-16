import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms

# ---------------- MODEL ----------------
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 56 * 56, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        return self.fc(self.conv(x))


# load model
model = CNN()
model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()

# ---------------- UI ----------------
st.title("Aerial Image Classification (PyTorch CNN)")

upload = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

labels = ["Class 0", "Class 1"]  # change this

if upload:
    img = Image.open(upload).convert("RGB")
    st.image(img, caption="Uploaded Image")

    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(img)
        pred = torch.argmax(output, 1).item()

    st.success(f"Prediction: {labels[pred]}")
