
import streamlit as st
import numpy as np
import uuid
from PIL import Image
from io import BytesIO
import requests
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Load model
model = load_model("model_transfer_learning.h5")

# Class labels
class_labels = ['Apple', 'Banana', 'Grape', 'Mango', 'Strawberry']
threshold = 0.80  # confidence threshold
gap_threshold = 0.15  # min difference between top-1 and top-2

st.set_page_config(page_title="Fruit Classifier", layout="centered")
st.title("🍓 Fruit Classifier - MobileNetV2")

if "history" not in st.session_state:
    st.session_state.history = []

method = st.radio("Select image input method:", ["Upload Image", "Image URL", "Use Camera"])
img = None

if method == "Upload Image":
    uploaded = st.file_uploader("Upload a fruit image...", type=["jpg", "jpeg", "png", "webp"])
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
elif method == "Image URL":
    url = st.text_input("Enter direct image URL (must end in .jpg/.png):")
    if url:
        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            res = requests.get(url, headers=headers)
            img = Image.open(BytesIO(res.content)).convert("RGB")
        except:
            st.error("❌ Unable to load image.")
elif method == "Use Camera":
    camera_img = st.camera_input("Take a photo")
    if camera_img:
        img = Image.open(camera_img).convert("RGB")

if img:
    st.image(img, caption="Input Image", use_container_width=True)
    img_resized = img.resize((224, 224))
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    preds = model.predict(img_array)
    sorted_indices = np.argsort(preds[0])[::-1]
    top1_idx, top2_idx = sorted_indices[:2]
    top1_conf = preds[0][top1_idx]
    top2_conf = preds[0][top2_idx]

    st.markdown("### 🧠 Prediction Result")
    if top1_conf >= threshold and (top1_conf - top2_conf) >= gap_threshold:
        pred_class = class_labels[top1_idx]
        st.success(f"**{pred_class}** ({top1_conf*100:.2f}%)")
        display_class = pred_class
    else:
        st.warning(f"**Unknown / Not a Fruit** ({top1_conf*100:.2f}%)")
        display_class = "Unknown"

    st.session_state.history.append({
        "class": display_class,
        "confidence": f"{top1_conf*100:.2f}%",
        "id": str(uuid.uuid4())[:6]
    })

st.sidebar.title("📈 Prediction History")
if st.session_state.history:
    for item in reversed(st.session_state.history[-5:]):
        st.sidebar.markdown(f"- **{item['class']}** ({item['confidence']})")
else:
    st.sidebar.write("No predictions yet.")
