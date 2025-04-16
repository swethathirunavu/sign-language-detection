import cv2
import mediapipe as mp

class handDetector:
    def __init__(self, mode=False, maxHands=1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=self.mode,
                                        max_num_hands=self.maxHands,
                                        min_detection_confidence=self.detectionCon,
                                        min_tracking_confidence=self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        hand_list = []
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
                hand_list.append(handLms)
        return hand_list

    def getBoundingBox(self, handLms):
        h, w, _ = 480, 640, 3
        xList, yList = [], []
        for lm in handLms.landmark:
            xList.append(int(lm.x * w))
            yList.append(int(lm.y * h))
        x_min, x_max = min(xList), max(xList)
        y_min, y_max = min(yList), max(yList)
        return x_min, y_min, x_max - x_min, y_max - y_min




