import cv2
import mediapipe as mp
from collections import deque
import numpy as np

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

    # checking if right hand index finger visible
    if results.right_hand_landmarks:
        index_finger_tip = results.right_hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

        # Get the normalized coordinates
        h, w, _ = frame.shape
        cx, cy = int(index_finger_tip.x * w), int(index_finger_tip.y * h)

        tracking_index.appendleft((cx, cy))

    for i in range (1, len(tracking_index)):
        if tracking_index[i - 1] is None or tracking_index[i] is None:
            continue
        thick = int(np.sqrt(len(tracking_index) / float(i + 1)) * 4.5)
        cv2.line(image, tracking_index[i - 1], tracking_index[i], (255, 0, 0), thick)

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

def getCoords(results, frame):
    pass
