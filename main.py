import sys
import os
import cv2
import time
import streamlit as st

# Add the current directory to sys.path to ensure HandTrackingModule can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import handtrackingmodule as htm  # Import the hand tracking module

# Set up the Streamlit page
st.title("Sign Language Detection")
st.sidebar.image("sign-language-alphabet.png", caption="Sign Language Alphabet", use_column_width=True)

# Initialize camera
hCam, wCam = 480, 640
cap = cv2.VideoCapture(0)
cap.set(4, hCam)
cap.set(3, wCam)

detector = htm.handDetector(detectionCon=0)

# Streamlit image display
image_placeholder = st.empty()

# Streamlit text display for result
sentence_placeholder = st.empty()

# Function to process frames and display results
def process_frame():
    success, img = cap.read()
    img = detector.findHands(img)
    posList = detector.findPosition(img, draw=False)

    result = ""
    fingers = []
    
    # Finger positions
    finger_mcp = [5, 9, 13, 17]
    finger_dip = [6, 10, 14, 18]
    finger_pip = [7, 11, 15, 19]
    finger_tip = [8, 12, 16, 20]
    
    for id in range(4):
        if posList[finger_tip[id]][1] + 25 < posList[finger_dip[id]][1] and posList[16][2] < posList[20][2]:
            fingers.append(0.25)
        elif posList[finger_tip[id]][2] > posList[finger_dip[id]][2]:
            fingers.append(0)
        elif posList[finger_tip[id]][2] < posList[finger_pip[id]][2]:
            fingers.append(1)
        elif posList[finger_tip[id]][1] > posList[finger_pip[id]][1] and posList[finger_tip[id]][1] > posList[finger_dip[id]][1]:
            fingers.append(0.5)

    # Detecting letters based on finger positions and hand landmarks
    if posList[3][2] > posList[4][2] and posList[3][1] > posList[6][1] and posList[4][2] < posList[6][2] and fingers.count(0) == 4:
        result = "A"
    elif posList[3][1] > posList[4][1] and fingers.count(1) == 4:
        result = "B"
    elif posList[3][1] > posList[4][1] and fingers.count(1) == 3:
        result = "C"
    elif posList[3][2] < posList[4][2] and posList[3][2] < posList[7][2]:
        result = "D"
    elif posList[3][2] < posList[4][2] and posList[3][1] > posList[4][1] and posList[4][2] < posList[7][2]:
        result = "E"
    elif posList[3][1] > posList[4][1] and posList[16][2] < posList[20][2]:
        result = "F"
    elif posList[3][2] > posList[4][2] and posList[4][2] < posList[6][2] and posList[3][2] < posList[7][2]:
        result = "G"
    elif posList[3][2] < posList[4][2] and posList[3][1] > posList[4][1] and posList[4][2] < posList[7][2]:
        result = "H"
    elif posList[4][1] < posList[5][1] and posList[3][2] > posList[4][2]:
        result = "I"
    elif posList[5][1] > posList[6][1] and posList[3][2] < posList[7][2]:
        result = "J"
    elif posList[3][2] > posList[4][2] and posList[3][2] < posList[5][2]:
        result = "K"
    elif posList[3][1] > posList[4][1] and posList[2][2] < posList[6][2]:
        result = "L"
    elif posList[4][1] < posList[5][1] and posList[6][2] < posList[7][2]:
        result = "M"
    elif posList[2][2] < posList[4][2] and posList[3][2] > posList[4][2]:
        result = "N"
    elif posList[2][2] > posList[3][2] and posList[4][2] < posList[6][2]:
        result = "O"
    elif posList[2][2] < posList[3][2] and posList[7][2] > posList[6][2]:
        result = "P"
    elif posList[3][1] < posList[4][1] and posList[2][2] < posList[6][2]:
        result = "Q"
    elif posList[5][1] < posList[6][1] and posList[2][2] < posList[4][2]:
        result = "R"
    elif posList[3][1] < posList[5][1] and posList[5][1] < posList[6][1]:
        result = "S"
    elif posList[2][2] > posList[4][2] and posList[6][2] > posList[5][2]:
        result = "T"
    elif posList[5][2] > posList[6][2] and posList[2][1] < posList[6][1]:
        result = "U"
    elif posList[4][1] > posList[6][1] and posList[2][2] < posList[5][2]:
        result = "V"
    elif posList[6][2] < posList[7][2] and posList[4][2] > posList[6][2]:
        result = "W"
    elif posList[2][1] > posList[6][2] and posList[3][2] > posList[5][2]:
        result = "X"
    elif posList[2][2] < posList[3][2] and posList[5][2] < posList[6][2]:
        result = "Y"
    elif posList[6][1] < posList[5][2] and posList[2][2] < posList[4][2]:
        result = "Z"
    
    # Display image in Streamlit
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image_placeholder.image(img_rgb, channels="RGB", use_column_width=True)

    # Display the result (detected letter)
    sentence_placeholder.write(f"Detected Letter: {result}")

# Loop to continuously capture frames
while True:
    process_frame()
    time.sleep(0.1)
