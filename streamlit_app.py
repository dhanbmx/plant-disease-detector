import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image

st.set_page_config(page_title="Plant Disease Detector", page_icon="🌿")

@st.cache_resource
def load():
    model = tf.keras.models.load_model("plant_model.keras")
    names = json.load(open("class_names.json"))
    return model, names

model, class_names = load()

def clean(name):
    return name.replace("___", " - ").replace("_", " ")

st.title("🌿 Plant Disease Detector")
st.write(
    "Trained on PlantVillage (38 classes) with MobileNetV2 transfer learning. "
    "Held-out accuracy: 95.5%. Works best on a clear, close-up photo of one leaf. "
    "Not a substitute for expert diagnosis."
)

file = st.file_uploader("Upload a leaf photo", type=["jpg", "jpeg", "png"])
cam = st.camera_input("...or take a photo")
src = file or cam

if src:
    img = Image.open(src).convert("RGB")
    st.image(img, use_container_width=True)
    arr = np.array(img.resize((224, 224)), dtype="float32")
    probs = model.predict(arr[None, ...], verbose=0)[0]
    top = probs.argsort()[-3:][::-1]
    st.subheader("Prediction")
    for i in top:
        st.write(f"**{clean(class_names[i])}**: {probs[i]*100:.1f}%")
        st.progress(float(probs[i]))
