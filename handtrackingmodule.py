import cv2
import mediapipe as mp

class HandDetector:
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(min_detection_confidence, min_tracking_confidence)
        self.mp_draw = mp.solutions.drawing_utils

    def find_hands(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)
        
        if self.results.multi_hand_landmarks:
            for landmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(img, landmarks, self.mp_hands.HAND_CONNECTIONS)
        return img

    def find_position(self, img, hand_no=0, draw=True):
        lm_list = []
        if self.results.multi_hand_landmarks:
            hand = self.results.multi_hand_landmarks[hand_no]
            for id, landmark in enumerate(hand.landmark):
                h, w, c = img.shape
                lm_list.append([id, int(landmark.x * w), int(landmark.y * h)])
                if draw:
                    cv2.circle(img, (lm_list[id][1], lm_list[id][2]), 5, (0, 255, 0), cv2.FILLED)
        return lm_list
