import streamlit as st
import torch
from torchvision.models import vit_b_16
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms

# Model setup
num_classes = 4
model = vit_b_16(weights=None)
model.heads = nn.Linear(768, num_classes)

model.load_state_dict(torch.load("final_model.pth", map_location="cpu"))
model.eval()

classes = ["Glioma", "Meningioma", "Pituitary", "No Tumor"]

st.title("Brain Tumor Classification (ViT + SSL)")

file = st.file_uploader("Upload MRI Image", type=["jpg", "png"])

if file:
    img = Image.open(file).convert("RGB")
    st.image(img, caption="Uploaded Image")

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])

    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(img)
        probs = torch.softmax(output, dim=1)

    pred = torch.argmax(probs, dim=1).item()
    confidence = probs[0][pred].item()

    st.success(f"Prediction: {classes[pred]}")
    st.info(f"Confidence: {confidence*100:.2f}%")