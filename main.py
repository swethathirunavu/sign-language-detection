import cv2
import mediapipe as mp
import streamlit as st
import numpy as np
from PIL import Image


st.set_page_config(layout="wide", page_title="Sign Language Detector", page_icon="🤟")


st.title("🤟 Real-Time Sign Language Detection (A-Z)")


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.8)
mp_draw = mp.solutions.drawing_utils


sign_instructions = {
    "A": "Fist with thumb alongside the fingers.",
    "B": "Fingers straight up, thumb across palm.",
    "C": "Hand forms the shape of the letter 'C'.",
    "D": "Index up, other fingers touching thumb.",
    "E": "Fingers curled, thumb under fingers.",
    "F": "Thumb and index form circle, others up.",
    "G": "Thumb and index point sideways.",
    "H": "Index and middle point sideways.",
    "I": "Pinky finger up, others folded.",
    "J": "Draw a 'J' with pinky finger.",
    "K": "Index and middle up, thumb in between.",
    "L": "Index and thumb form 'L' shape.",
    "M": "Thumb under three fingers.",
    "N": "Thumb under two fingers.",
    "O": "All fingertips touch to form 'O'.",
    "P": "Like 'K' but tilted down.",
    "Q": "Like 'G' but tilted down.",
    "R": "Index crossed over middle.",
    "S": "Fist with thumb in front.",
    "T": "Thumb between index and middle.",
    "U": "Index and middle together.",
    "V": "Index and middle spread.",
    "W": "3 fingers up - index, middle, ring.",
    "X": "Bent index finger (like hook).",
    "Y": "Thumb and pinky out (hang loose).",
    "Z": "Draw 'Z' shape with index in air."
}


def detect_sign(handLms, img):
    result = ""
    posList = []

    for id, lm in enumerate(handLms.landmark):
        h, w, _ = img.shape
        cx, cy = int(lm.x * w), int(lm.y * h)
        posList.append([id, cx, cy])

    fingers = []
    finger_mcp = [5,9,13,17]
    finger_dip = [6,10,14,18]
    finger_pip = [7,11,15,19]
    finger_tip = [8,12,16,20]

    for id in range(4):
        if(posList[finger_tip[id]][1]+ 25  < posList[finger_dip[id]][1] and posList[16][2]<posList[20][2]):
            fingers.append(0.25)
        elif(posList[finger_tip[id]][2] > posList[finger_dip[id]][2]):
            fingers.append(0)
        elif(posList[finger_tip[id]][2] < posList[finger_pip[id]][2]): 
            fingers.append(1)
        elif(posList[finger_tip[id]][1] > posList[finger_pip[id]][1] and posList[finger_tip[id]][1] > posList[finger_dip[id]][1]): 
            fingers.append(0.5)

    if len(posList) != 21:
        return ""

    # Letter detection based on your logic
    if(posList[3][2] > posList[4][2]) and (posList[3][1] > posList[6][1])and (posList[4][2] < posList[6][2]) and fingers.count(0) == 4:
        result = "A"
    elif(posList[3][1] > posList[4][1]) and fingers.count(1) == 4:
        result = "B"
    elif(posList[3][1] > posList[6][1]) and fingers.count(0.5) >= 1 and (posList[4][2]> posList[8][2]):
        result = "C"
    elif(fingers[0]==1) and fingers.count(0) == 3 and (posList[3][1] > posList[4][1]):
        result = "D"
    elif (posList[3][1] < posList[6][1]) and fingers.count(0) == 4 and posList[12][2]<posList[4][2]:
        result = "E"
    elif (fingers.count(1) == 3) and (fingers[0]==0) and (posList[3][2] > posList[4][2]):
        result = "F"
    elif(fingers[0]==0.25) and fingers.count(0) == 3:
        result = "G"
    elif(fingers[0]==0.25) and(fingers[1]==0.25) and fingers.count(0) == 2:
        result = "H"
    elif (posList[4][1] < posList[6][1]) and fingers.count(0) == 3:
        if (len(fingers)==4 and fingers[3] == 1):
            result = "I"
    elif (posList[4][1] < posList[6][1] and posList[4][1] > posList[10][1] and fingers.count(1) == 2):
        result = "K"
    elif(fingers[0]==1) and fingers.count(0) == 3 and (posList[3][1] < posList[4][1]):
        result = "L"
    elif (posList[4][1] < posList[16][1]) and fingers.count(0) == 4:
        result = "M"
    elif (posList[4][1] < posList[12][1]) and fingers.count(0) == 4:
        result = "N"
    elif(posList[4][2] < posList[8][2]) and (posList[4][2] < posList[12][2]) and (posList[4][2] < posList[16][2]) and (posList[4][2] < posList[20][2]):
        result = "O"
    elif(fingers[2] == 0)  and (posList[4][2] < posList[12][2]) and (posList[4][2] > posList[6][2]):
        if (len(fingers)==4 and fingers[3] == 0):
            result = "P"
    elif(fingers[1] == 0) and (fingers[2] == 0) and (fingers[3] == 0) and (posList[8][2] > posList[5][2]) and (posList[4][2] < posList[1][2]):
        result = "Q"
    elif(posList[8][1] < posList[12][1]) and (fingers.count(1) == 2) and (posList[9][1] > posList[4][1]):
        result = "R"
    elif (posList[4][1] > posList[12][1]) and posList[4][2]<posList[12][2] and fingers.count(0) == 4:
        result = "S"
    elif (posList[4][1] > posList[12][1]) and posList[4][2]<posList[6][2] and fingers.count(0) == 4:
        result = "T"
    elif (posList[4][1] < posList[6][1] and posList[4][1] < posList[10][1] and fingers.count(1) == 2 and posList[3][2] > posList[4][2] and (posList[8][1] - posList[11][1]) <= 50):
        result = "U"
    elif (posList[4][1] < posList[6][1] and posList[4][1] < posList[10][1] and fingers.count(1) == 2 and posList[3][2] > posList[4][2]):
        result = "V"
    elif (posList[4][1] < posList[6][1] and posList[4][1] < posList[10][1] and fingers.count(1) == 3):
        result = "W"
    elif (fingers[0] == 0.5 and fingers.count(0) == 3 and posList[4][1] > posList[6][1]):
        result = "X"
    elif(fingers.count(0) == 3) and (posList[3][1] < posList[4][1]):
        if (len(fingers)==4 and fingers[3] == 1):
            result = "Y"
    
    return result


if 'camera_on' not in st.session_state:
    st.session_state.camera_on = False
    
if 'formed_word' not in st.session_state:
    st.session_state.formed_word = ""
    
if 'last_letter' not in st.session_state:
    st.session_state.last_letter = ""
    
if 'letter_confirmed' not in st.session_state:
    st.session_state.letter_confirmed = False
    
if 'confirmation_timer' not in st.session_state:
    st.session_state.confirmation_timer = 0

if 'show_image' not in st.session_state:
    st.session_state.show_image = False
    
if 'show_instructions' not in st.session_state:
    st.session_state.show_instructions = False


with st.sidebar:
    st.header("Controls")
    
    
    st.subheader("Camera")
    cam_col1, cam_col2 = st.columns(2)
    with cam_col1:
        start_button = st.button("▶️ Start", key="start_btn")
    with cam_col2:
        stop_button = st.button("⏹️ Stop", key="stop_btn")
    
    
    st.subheader("Sentence")
    clear_button = st.button("🧹 Clear Sentence", key="clear_btn")
    
    
    st.subheader("Reference")
    ref_col1, ref_col2 = st.columns(2)
    with ref_col1:
        show_image = st.button(">", key="show_img", help="Show ASL reference chart")
    with ref_col2:
        show_instructions = st.button("<", key="show_inst", help="Show letter instructions")


if start_button:
    st.session_state.camera_on = True
    
if stop_button:
    st.session_state.camera_on = False
    
if clear_button:
    st.session_state.formed_word = ""
    
if show_image:
    st.session_state.show_image = True
    st.session_state.show_instructions = False
    
if show_instructions:
    st.session_state.show_instructions = True
    st.session_state.show_image = False


col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("📷 Live Webcam Feed")
    frame_placeholder = st.empty()

with col2:
    
    st.subheader("📝 Detected Letter")
    result_text = st.empty()
    
    
    st.subheader("🔤 Sentence Formation")
    st.info("Show sign 'B' for adding a space between words")
    word_placeholder = st.empty()
    
    
    if st.session_state.show_image:
        st.subheader("📸 ASL Reference Chart")
        st.image("sign-language-alphabet.png", use_container_width=True)
    
    if st.session_state.show_instructions:
        st.subheader("📖 Sign Instructions")
        for letter, desc in sign_instructions.items():
            st.write(f"**{letter}**: {desc}")


cap = cv2.VideoCapture(0)


def process_webcam():
    letter_display_time = 20  
    
    while st.session_state.camera_on:
        success, img = cap.read()
        if not success:
            st.error("Camera not accessible.")
            break

        img = cv2.flip(img, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        current_result = ""

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)
                current_result = detect_sign(handLms, img)

        
        if current_result:
            if current_result == st.session_state.last_letter:
                st.session_state.confirmation_timer += 1
                
                
                if st.session_state.confirmation_timer >= letter_display_time and not st.session_state.letter_confirmed:
                    
                    if current_result == "B":
                        st.session_state.formed_word += " "
                    else:
                        st.session_state.formed_word += current_result
                    
                    st.session_state.letter_confirmed = True
                    
                    word_placeholder.markdown(f"## {st.session_state.formed_word}")
            else:
                
                st.session_state.last_letter = current_result
                st.session_state.confirmation_timer = 0
                st.session_state.letter_confirmed = False
        else:
            
            st.session_state.last_letter = ""
            st.session_state.confirmation_timer = 0
            st.session_state.letter_confirmed = False

        
        cv2.rectangle(img, (28,255), (178, 425), (0, 225, 0), cv2.FILLED)
        cv2.putText(img, str(current_result), (55,400), cv2.FONT_HERSHEY_COMPLEX, 5, (255,0,0), 15)
        
        
        if st.session_state.last_letter and st.session_state.confirmation_timer > 0:
            progress = min(1.0, st.session_state.confirmation_timer / letter_display_time)
            cv2.rectangle(img, (30, 430), (30 + int(150 * progress), 460), (0, 255, 255), cv2.FILLED)
        
        
        frame_placeholder.image(img, channels="RGB")
        result_text.markdown(f"## 👉 Letter: {current_result}")
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break


if st.session_state.camera_on:
    process_webcam()
else:
    
    frame_placeholder.image("https://place-hold.it/640x480?text=Camera%20Off&fontsize=32", use_container_width=True)
    

if 'cap' in locals():
    cap.release()
