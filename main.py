import cv2
import os
import mediapipe as mp

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

capture = cv2.VideoCapture(0);
while capture.isOpened():
    ret, frame = capture.read()
    frame = cv2.resize(frame, (1000, 750))
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.flip(frame, 1)

    # Making predictions using holistic model
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    results = holistic_model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Drawing the Facial Landmarks
    # mp_drawing.draw_landmarks(
    #   image,
    #   results.face_landmarks,
    #   mp_holistic.FACEMESH_CONTOURS,
    #   mp_drawing.DrawingSpec(
    #     color=(255,0,255),
    #     thickness=1,
    #     circle_radius=1
    #   ),
    #   mp_drawing.DrawingSpec(
    #     color=(0,255,255),
    #     thickness=1,
    #     circle_radius=1
    #   )
    # )

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

    # Display the resulting image
    cv2.imshow("Facial and Hand Landmarks", image)

    # Enter key 'q' to break the loop
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# When all the process is done
# Release the capture and destroy all windows
capture.release()
cv2.destroyAllWindows()
