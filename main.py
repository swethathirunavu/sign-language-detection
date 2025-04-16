import cv2
import numpy as np
import streamlit as st
from handtrackingmodule import handDetector

def process_frame():
    # Create a placeholder for the video feed
    frame_placeholder = st.empty()
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    # Check if webcam is opened successfully
    if not cap.isOpened():
        st.error("Could not open webcam. Please check your camera connection.")
        return
    
    # Initialize the hand detector
    detector = handDetector(detectionCon=0.7)
    
    # Set the frame width and height
    cap.set(3, 640)
    cap.set(4, 480)
    
    while True:
        # Read frame from webcam
        success, img = cap.read()
        
        # Check if frame is valid
        if not success or img is None:
            st.error("Failed to receive frame from webcam.")
            break
            
        try:
            # Find hands in the image
            img = detector.findHands(img)
            
            # Get positions of hand landmarks
            lmList = detector.findPosition(img, draw=False)
            
            # Process the landmarks for sign language detection
            if lmList:
                # Add your sign language detection logic here
                # For example, display the position of index finger
                if len(lmList) > 8:  # Check if index finger tip is detected
                    x, y = lmList[8][1], lmList[8][2]
                    cv2.circle(img, (x, y), 15, (0, 255, 255), cv2.FILLED)
            
            # Convert color from BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Display the frame
            frame_placeholder.image(img, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error processing frame: {str(e)}")
            break
            
    # Release the webcam
    cap.release()

def main():
    st.title("Sign Language Detection")
    st.write("This application detects hand movements for sign language.")
    
    # Start button
    start = st.button("Start Detection")
    
    if start:
        process_frame()

if __name__ == "__main__":
    main()
