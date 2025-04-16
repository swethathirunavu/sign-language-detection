import cv2
import mediapipe as mp
import numpy as np

class handDetector():
    def __init__(self, detectionCon=0.5):
        self.detectionCon = detectionCon
        # Initialize MediaPipe Hands module
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(min_detection_confidence=self.detectionCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.results = None
    
    def findHands(self, img, draw=True):
        """Detect hands and draw landmarks"""
        # Check if image is valid
        if img is None or not isinstance(img, np.ndarray):
            print("Invalid image input")
            return img
        
        try:
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.results = self.hands.process(imgRGB)
            
            # If hands are detected
            if self.results.multi_hand_landmarks:
                for handLandmarks in self.results.multi_hand_landmarks:
                    if draw:
                        self.mpDraw.draw_landmarks(img, handLandmarks, self.mpHands.HAND_CONNECTIONS)
            return img
        except Exception as e:
            print(f"Error in findHands: {e}")
            return img
    
    def findPosition(self, img, handNo=0, draw=True):
        """Return a list of landmark positions for a specific hand"""
        lmList = []
        if img is None or not isinstance(img, np.ndarray):
            return lmList
            
        try:
            if self.results and self.results.multi_hand_landmarks:
                if handNo < len(self.results.multi_hand_landmarks):
                    myHand = self.results.multi_hand_landmarks[handNo]
                    for id, lm in enumerate(myHand.landmark):
                        h, w, c = img.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        lmList.append([id, cx, cy])
                        if draw:
                            cv2.circle(img, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
            return lmList
        except Exception as e:
            print(f"Error in findPosition: {e}")
            return lmList
