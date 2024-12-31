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
        if mode == 1: # right hand index finger
            point = results.right_hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        elif mode == 2: # right hand thumb
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

def thumbIndexDetected(image, distance_threshold, indexCoords, thumbCoords):
    distance = math.sqrt((indexCoords[0] - thumbCoords[0]) ** 2 + (indexCoords[1] - thumbCoords[1]) ** 2)
    if distance < distance_threshold:
        # Get the bounding box for the thumb and index finger
        top_left = (min(thumbCoords[0], indexCoords[0]), min(thumbCoords[1], indexCoords[1]))
        bottom_right = (max(thumbCoords[0], indexCoords[0]), max(thumbCoords[1], indexCoords[1]))

        # Increase the size of the box by a margin (expand the box to cover more area)
        margin = 75
        top_left = (top_left[0] - margin, top_left[1] - margin)
        bottom_right = (bottom_right[0] + margin, bottom_right[1] + margin)

        cv2.rectangle(image, top_left, bottom_right, (0, 0, 255), 2)

def calculate_distance(landmark1, landmark2, frame):
    h, w, _ = frame.shape
    x1, y1 = int(landmark1.x * w), int(landmark1.y * h)
    x2, y2 = int(landmark2.x * w), int(landmark2.y * h)
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

# Function to detect fist
def fistDetected(hand_landmarks, frame, image):
    # List of fingertips and their respective bases
    fingertip_indices = [
        mp_hands.HandLandmark.INDEX_FINGER_TIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
        mp_hands.HandLandmark.RING_FINGER_TIP,
        mp_hands.HandLandmark.PINKY_TIP,
    ]
    finger_base_indices = [
        mp_hands.HandLandmark.INDEX_FINGER_MCP,
        mp_hands.HandLandmark.MIDDLE_FINGER_MCP,
        mp_hands.HandLandmark.RING_FINGER_MCP,
        mp_hands.HandLandmark.PINKY_MCP,
    ]

    # Threshold for distance (adjust based on frame size or calibration)
    threshold = 30  # Pixels, depends on video resolution

    for tip, base in zip(fingertip_indices, finger_base_indices):
        distance = calculate_distance(
            hand_landmarks.landmark[tip], hand_landmarks.landmark[base], frame
        )
        if distance > threshold:
            return False  # If any finger is extended, it's not a fist

    # Draw green box to indicate fist
    h, w, _ = frame.shape
    margin = 20
    # Get all x and y coordinates of landmarks
    x_coords = [int(landmark.x * w) for landmark in hand_landmarks.landmark]
    y_coords = [int(landmark.y * h) for landmark in hand_landmarks.landmark]
    # Find the min and max of the coordinates
    x_min, x_max = max(0, min(x_coords) - margin), min(w, max(x_coords) + margin)
    y_min, y_max = max(0, min(y_coords) - margin), min(h, max(y_coords) + margin)

    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)
    return True

# Grabbing the Holistic Model from Mediapipe and
# Initializing the Model
mp_holistic = mp.solutions.holistic
holistic_model = mp_holistic.Holistic(
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Initializing the drawing utils for drawing the facial landmarks on image
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hand_landmark_drawing_spec = mp_drawing.DrawingSpec(thickness=5, circle_radius=5)
hand_connection_drawing_spec = mp_drawing.DrawingSpec(thickness=10, circle_radius=10)

# tracking feature
tracking_index = deque(maxlen=40)
tracking_thumb = deque(maxlen=40)
# distance between thumb and index finger
distance_threshold = 35

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
        thumbIndexDetected(image, distance_threshold, indexCoords, thumbCoords)

    # HOW TO DETECT FIST
    if results.right_hand_landmarks:
        fistDetected(results.right_hand_landmarks, frame, image)

    # Optional finger tracking
    # drawTracking(image, tracking_index)
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

