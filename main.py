import cv2
import mediapipe as mp
from collections import deque

import os

# Grabbing the Holistic Model from Mediapipe and
# Initializing the Model
mp_holistic = mp.solutions.holistic
holistic_model = mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Initializing the drawing utils for drawing the facial landmarks on image
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hand_landmark_drawing_spec = mp_drawing.DrawingSpec(thickness=5, circle_radius=5)
hand_connection_drawing_spec = mp_drawing.DrawingSpec(thickness=10, circle_radius=10)

# tracking feature
tracking_index = deque(maxlen=50)
prev_x, prev_y = None, None
# Create a blank canvas for drawing
canvas = None

capture = cv2.VideoCapture(0);
while capture.isOpened():
    ret, frame = capture.read()

    if not ret:
        break
   
    frame = cv2.resize(frame, (1000, 750))

    if canvas is None:
        canvas = frame.copy() * 0

     # frame = cv2.flip(frame, 1)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Making predictions using holistic model
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    results = holistic_model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Drawing Left hand Land Marks
    mp_drawing.draw_landmarks(
      image,
      results.left_hand_landmarks,
      mp_holistic.HAND_CONNECTIONS
    )

    # Drawing right hand landmarks
    mp_drawing.draw_landmarks(
      image,
      results.right_hand_landmarks,
      mp_holistic.HAND_CONNECTIONS
    )

    # checking if right hand index finger visible
    if results.right_hand_landmarks:
        index_finger_tip = results.right_hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        if 0 <= index_finger_tip.x <= 1 and 0 <= index_finger_tip.y <= 1:
            print("Index finger of right hand is visible at:", index_finger_tip.x, index_finger_tip.y)

        # Get the normalized coordinates
        h, w, _ = frame.shape
        cx, cy = int(index_finger_tip.x * w), int(index_finger_tip.y * h)

        # tracking_index.appendleft((cx, cy))
        # cv2.line(image, (prev_x, prev_y), (cx, cy), (255, 0, 0), thickness=5)

        # Draw a line from the previous position to the current position
        if prev_x is not None and prev_y is not None:
            cv2.line(image, (prev_x, prev_y), (cx, cy), (255, 0, 0), thickness=5)

        prev_x, prev_y = cx, cy
    else:
        prev_x, prev_y = None, None

    # Display the resulting frame
    image = cv2.flip(image, 1)
    cv2.imshow("GestureFlow", image)

    # Enter key 'q' to break the loop
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# When all the process is done
# Release the capture and destroy all windows
capture.release()
cv2.destroyAllWindows()
