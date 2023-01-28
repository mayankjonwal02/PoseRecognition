import cv2
from cv2 import waitKey
import mediapipe as mp

mp_drawing=mp.solutions.drawing_utils
mp_holistic=mp.solutions.holistic
mp_holistic1=mp_holistic.Holistic(min_detection_confidence=0,min_tracking_confidence=0.5)

cap=cv2.VideoCapture(0)
while cap.isOpened():
    _,frame=cap.read()
    #frame=cv2.flip(frame,1)
    frame1=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    output=mp_holistic1.process(frame1)
    
    mp_drawing.draw_landmarks(image=frame,landmark_list=output.right_hand_landmarks,connections=mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image=frame,landmark_list=output.left_hand_landmarks,connections=mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image=frame,landmark_list=output.pose_landmarks,connections=mp_holistic.POSE_CONNECTIONS)
    #mp_drawing.draw_landmarks(frame,output.face_landmarks,mp_holistic.FACE_CONNECTIONS)





    cv2.imshow("Holistic detection",cv2.flip(frame,1))
    waitKey(1)