import cv2
import streamlit as st
from handtrackingmodule import HandDetector

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Initialize HandDetector
detector = HandDetector()

# Streamlit UI setup
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
    
    # Convert to RGB (Streamlit expects RGB)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Display the frame in Streamlit
    st.image(frame_rgb, channels="RGB", use_column_width=True)

    # Stop the webcam when Streamlit is done
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam
cap.release()
cv2.destroyAllWindows()
