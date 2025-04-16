import streamlit as st
import cv2
import numpy as np
import time
from PIL import Image
import handtrackingmodule as htm

st.set_page_config(page_title="Sign Language Detection", layout="wide")

st.title("🤟 Real-Time Sign Language Detection")
st.image("sign-language-alphabet.png", caption="Sign Language Reference", use_container_width=True)

# Load model
import joblib
model = joblib.load("model/sign_model.pkl")

# Initialize hand detector
detector = htm.handDetector()

# Access webcam
cap = cv2.VideoCapture(0)

stframe = st.empty()
predicted_label = st.empty()

while True:
    success, img = cap.read()
    if not success:
        st.error("❌ Failed to access webcam")
        break

    img = cv2.flip(img, 1)
    hands = detector.findHands(img)
    if hands:
        x, y, w, h = detector.getBoundingBox(hands[0])
        hand_img = img[y:y+h, x:x+w]
        try:
            resized_img = cv2.resize(hand_img, (64, 64))
            norm_img = resized_img / 255.0
            input_img = norm_img.reshape(1, 64, 64, 3)

            prediction = model.predict(input_img)
            letter = chr(prediction[0] + 65)

            predicted_label.markdown(f"### 🧠 Prediction: **{letter}**")
        except Exception as e:
            predicted_label.warning("⚠️ Unable to process the hand image.")
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    stframe.image(img_rgb, channels="RGB", use_container_width=True)

    # Optional: slow down for clarity
    time.sleep(0.03)


