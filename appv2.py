import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2
import numpy as np
import tempfile
import time

# -----------------------------
# 1️⃣ Page configuration
# -----------------------------
st.set_page_config(
    page_title="DeepCheck – Deepfake Detection",
    page_icon="🧠",
    layout="wide",
)

# Sidebar info
st.sidebar.title("🧠 DeepCheck")
st.sidebar.markdown("""
**Deepfake Video Detection System**

Upload a short video clip, and our AI model will analyze
the first frame to classify it as **Real** or **Fake**.

📊 **Model:** ResNet18 (Fine-tuned)  
🎯 **Classes:** Real / Fake  
""")

# -----------------------------
# 2️⃣ Load the model
# -----------------------------
@st.cache_resource
def load_model():
    model = models.resnet18(pretrained=False)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)  # Real/Fake
    state_dict = torch.load("weights_finertuned.pth", map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    model.eval()
    return model

model = load_model()

# -----------------------------
# 3️⃣ Preprocessing
# -----------------------------
def preprocess_image(img):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    img = Image.fromarray(img)
    img = transform(img).unsqueeze(0)
    return img

# -----------------------------
# 4️⃣ Prediction
# -----------------------------
def predict(img):
    img_tensor = preprocess_image(img)
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, pred = torch.max(probabilities, 1)
    label = "Real" if pred.item() == 1 else "Fake"
    return label, confidence.item()

# -----------------------------
# 5️⃣ Main UI
# -----------------------------
st.title("🎥 DeepCheck – Deepfake Video Detection")
st.markdown(
    "<p style='color:gray;'>Upload a short video file to verify its authenticity using AI.</p>",
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader("📂 Upload a video file", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_path = temp_file.name

    cap = cv2.VideoCapture(temp_path)
    ret, frame = cap.read()

    if ret:
        # Display video preview
        st.image(frame[..., ::-1], caption="🎞 First frame from uploaded video", use_container_width=True)

        with st.spinner("Analyzing frame..."):
            time.sleep(1.5)
            label, confidence = predict(frame)

        # Display result with style
        if label == "Fake":
            st.markdown(
                f"<div style='background-color:#ffdddd;padding:20px;border-radius:10px;'>"
                f"<h3>🚫 Deepfake Detected</h3>"
                f"<p><b>Confidence:</b> {confidence*100:.2f}%</p></div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div style='background-color:#ddffdd;padding:20px;border-radius:10px;'>"
                f"<h3>✅ Real Video</h3>"
                f"<p><b>Confidence:</b> {confidence*100:.2f}%</p></div>",
                unsafe_allow_html=True,
            )
    else:
        st.error("⚠️ Could not read video file. Please upload a valid video.")

    cap.release()
