import cv2
import mediapipe as mp
from collections import deque
import numpy as np
import math

# this is for tracking tho
# it returns coords which get used to draw lines
# MODES -> 1 (right index) // 2 (left index) // 3 (right thumb)
def getCoords(cond, mode, frame):
    # checking if right hand index finger visible
    if cond:
        if mode == 1:
            point = results.right_hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        elif mode == 2:
            point = results.right_hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
        else:
            return None

        # Get the normalized coordinates
        h, w, _ = frame.shape
        cx, cy = int(point.x * w), int(point.y * h)
        return ((cx, cy))
    else:
        return None

def drawTracking(image, tracking):
    for i in range (1, len(tracking)):
        if tracking[i - 1] is None or tracking[i] is None:
            continue
        thick = int(np.sqrt(len(tracking) / float(i + 1)) * 4.5)
        cv2.line(image, tracking[i - 1], tracking[i], (255, 0, 0), thick)

# Grabbing the Holistic Model from Mediapipe and
# Initializing the Model
mp_holistic = mp.solutions.holistic
holistic_model = mp_holistic.Holistic(
    model_complexity=2,
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
tracking_thumb = deque(maxlen=50)
distance_threshold = 50

capture = cv2.VideoCapture(0);
while capture.isOpened():
    ret, frame = capture.read()
    if not ret:
        break
   
    frame = cv2.resize(frame, (800, 600))
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Making predictions using holistic model
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    results = holistic_model.process(image)
    # image.flags.writeable = True
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

    indexCoords = getCoords(results.right_hand_landmarks, 1, frame)
    if indexCoords is not None:
        tracking_index.appendleft(indexCoords)
    thumbCoords = getCoords(results.right_hand_landmarks, 2, frame)
    if thumbCoords is not None:
        tracking_thumb.appendleft(thumbCoords)

    # CHECK IF INDEX AND THUMB ARE CONNECTED
    if thumbCoords is not None and indexCoords is not None:
        distance = math.sqrt((indexCoords[0] - thumbCoords[0]) ** 2 + (indexCoords[1] - thumbCoords[1]) ** 2)
        if distance < distance_threshold:
            # Get the bounding box for the thumb and index finger
            top_left = (min(thumbCoords[0], indexCoords[0]), min(thumbCoords[1], indexCoords[1]))
            bottom_right = (max(thumbCoords[0], indexCoords[0]), max(thumbCoords[1], indexCoords[1]))

            # Increase the size of the box by a margin (expand the box to cover more area)
            margin = 75  # You can adjust this value to make the box bigger or smaller
            top_left = (top_left[0] - margin, top_left[1] - margin)
            bottom_right = (bottom_right[0] + margin, bottom_right[1] + margin)

            cv2.rectangle(image, top_left, bottom_right, (0, 0, 255), 2)

    # optional finger tracking
    drawTracking(image, tracking_index)
    # drawTracking(image, tracking_thumb)

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

