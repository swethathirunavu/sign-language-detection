import cv2
import numpy as np
import streamlit as st
from PIL import Image
import HandTrackingModule as htm  # Import the hand tracking module

# Set up the Streamlit app layout
st.title("Sign Language Recognition")

# Initialize the hand detector
detector = htm.handDetector(detectionCon=0.7)

# Set up a placeholder for webcam feed and output
video_placeholder = st.empty()
sentence_placeholder = st.empty()

# Camera input
camera_input = st.camera_input("Use your webcam")

# Check if camera input is available
if camera_input is not None:
    # Convert the Streamlit camera input to an OpenCV-compatible format
    img = Image.open(camera_input)
    img = np.array(img)

    # Process the image with the hand detector
    img = detector.findHands(img)
    posList = detector.findPosition(img, draw=False)

    # Process the detected hand positions to recognize gestures
    sentence = ""
    if len(posList) != 0:
        result = ""  # Define the result from gesture recognition

        # Add your gesture recognition logic here (for example, recognizing "A", "B", etc.)
        # Sample gesture recognition logic (replace this with your actual gesture detection code)
        if posList[4][1] < posList[6][1]:  # Example condition for recognizing "A"
            result = "A"
        elif posList[4][1] > posList[6][1]:  # Example condition for recognizing "B"
            result = "B"
        
        # Append the recognized letter or gesture to the sentence
        sentence += result

    # Display the recognized sentence below the video feed
    sentence_placeholder.text(f"Recognized Sentence: {sentence}")

    # Convert the image to RGB for Streamlit compatibility
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Display the webcam feed in Streamlit
    video_placeholder.image(img_rgb, channels="RGB", use_column_width=True)
else:
    st.warning("Please enable your camera.")

