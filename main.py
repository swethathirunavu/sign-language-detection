import cv2
import streamlit as st
from handtrackingmodule import HandDetector

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Initialize HandDetector
detector = HandDetector()

st.title("Sign Language Detection")

# Streamlit WebCam Stream
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Flip the frame horizontally for a mirror effect
    frame = cv2.flip(frame, 1)
    
    # Detect hands
    frame = detector.find_hands(frame)
    
    # Display the frame in Streamlit
    st.image(frame, channels="BGR", use_column_width=True)

    # Check if the user wants to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam
cap.release()
cv2.destroyAllWindows()
