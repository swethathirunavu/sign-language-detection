import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import string

# Streamlit config
st.set_page_config(page_title="Sign Language Detection", layout="centered")
st.title("🤟 Real-Time Sign Language Detection")
st.markdown("Press **Space** to add letter | **Backspace** to delete | **Enter** to clear.")

# Sentence state
if 'sentence' not in st.session_state:
    st.session_state.sentence = ""
if 'running' not in st.session_state:
    st.session_state.running = False

# Simulated labels A–Z
labels = list(string.ascii_uppercase)

# Simulated prediction (for demo)
def predict_dummy():
    return np.random.choice(labels)

# Start/Stop buttons
col1, col2 = st.columns(2)
with col1:
    if st.button("▶️ Start Detection"):
        st.session_state.running = True
with col2:
    if st.button("⏹ Stop Detection"):
        st.session_state.running = False

# Video and prediction placeholders
video_placeholder = st.empty()
pred_placeholder = st.empty()
sentence_placeholder = st.empty()

# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Main loop
if st.session_state.running:
    cap = cv2.VideoCapture(0)

    while st.session_state.running and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Couldn't access webcam")
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        predicted_char = ""

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                predicted_char = predict_dummy()

        # Show prediction only if hand detected
        if predicted_char:
            pred_placeholder.markdown(f"### 🖐 Detected Letter: `{predicted_char}`")
        else:
            pred_placeholder.markdown("### 🖐 Waiting for hand...")

        # Show camera
        video_placeholder.image(frame, channels="RGB")

        # Handle key input
        key = cv2.waitKey(1) & 0xFF
        if key == 32 and predicted_char:  # Space
            st.session_state.sentence += predicted_char
        elif key == 8:  # Backspace
            st.session_state.sentence = st.session_state.sentence[:-1]
        elif key == 13:  # Enter
            st.session_state.sentence = ""

        sentence_placeholder.markdown(f"**Current Sentence:** `{st.session_state.sentence}`")

    cap.release()
    cv2.destroyAllWindows()

# Final sentence display
if not st.session_state.running and st.session_state.sentence:
    st.markdown("---")
    st.subheader("📢 Final Sentence:")
    st.success(st.session_state.sentence)
