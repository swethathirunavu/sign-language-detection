import streamlit as st
import cv2
import numpy as np
from handtrackingmodule import HandDetector

st.title("Sign Language Hand Landmark Detector (Webcam)")

st.markdown("This app uses your webcam to detect hand landmarks in real-time.")

# Set up webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    st.error("Webcam not found!")
else:
    # Start the webcam stream
    stframe = st.empty()  # Placeholder for webcam feed in Streamlit

    # Initialize hand detector
    detector = HandDetector(maxHands=1)

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture frame from webcam.")
            break

        # Flip the frame horizontally for a later mirror effect
        frame = cv2.flip(frame, 1)

        # Detect hands and draw landmarks
        hands, frame = detector.findHands(frame, draw=True)

        # Display the frame with hand landmarks
        stframe.image(frame, channels="BGR", caption="Detected Hand Landmarks", use_column_width=True)

        # Exit condition for Streamlit app (press 'q' to exit)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam when done
    cap.release()
    cv2.destroyAllWindows()
