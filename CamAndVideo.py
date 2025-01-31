import numpy as np
import cv2
import mediapipe as mp

# load a video capture device
cap = cv2.VideoCapture(0)

# initialize mediapipe face detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

# while loop that will go until key is pressed
while True:
    # read a frame from the video capture device
    ret, frame = cap.read()

    # if the frame was not read correctly, break the loop
    if not ret:
        break

    # convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # process the frame for face detection
    results = face_detection.process(rgb_frame)

    # draw face detections on the frame
    if results.detections:
        for detection in results.detections:
            mp_drawing.draw_detection(frame, detection)

    # display the frame in a window
    cv2.imshow('frame', frame)

    # wait for 1 ms and check if the user pressed the 'q' key
    if cv2.waitKey(1) == ord('q'):
        break

# release the video capture device and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()