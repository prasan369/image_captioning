import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
import sys
sys.path.append('./src')
from model import ImageCaptioner

# ── LOAD MODEL ────────────────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    device = torch.device('cpu')
    checkpoint = torch.load('model.pth', map_location=device, weights_only=False)
    vocab = checkpoint['vocab']
    model = ImageCaptioner(embed_size=256, hidden_size=512, vocab_size=len(vocab))
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    return model, vocab, device

# ── IMAGE TRANSFORM ───────────────────────────────────────────────────────────

def transform_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# ── UI ────────────────────────────────────────────────────────────────────────

st.title("Image Captioning")
st.write("Upload an image and the model will generate a caption for it.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Generating caption..."):
        model, vocab, device = load_model()
        image_tensor = transform_image(image).to(device)
        caption = model.generate_caption(image_tensor, vocab, max_length=20)

    st.success(f"Caption: **{caption}**")