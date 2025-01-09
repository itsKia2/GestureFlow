import cv2
import mediapipe as mp
import numpy as np

def calculate_distance(landmark1, landmark2, frame):
    h, w, _ = frame.shape
    x1, y1 = int(landmark1.x * w), int(landmark1.y * h)
    x2, y2 = int(landmark2.x * w), int(landmark2.y * h)
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

# Grabbing the Holistic Model from Mediapipe and
# Initializing the Model
mp_holistic = mp.solutions.holistic
holistic_model = mp_holistic.Holistic(
    model_complexity=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

def displayText(image, text):
    # Set the font, size, and color
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    color = (255, 255, 255)  # White color for text
    thickness = 2
    # Get the dimensions of the text box
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    # Calculate the position for top-right alignment
    x = image.shape[1] - text_size[0] - 10  # Image width minus text width and some padding
    y = text_size[1] + 10  # Slight padding from the top
    # Add the text to the image
    cv2.putText(image, text, (x, y), font, font_scale, color, thickness)

previous_paper_contour = None
def findPaperWarp(frame):
    global previous_paper_contour
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Detect edges using Canny
    edges = cv2.Canny(blurred, 50, 150)
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize variables for the largest rectangular contour
    paper_contour = None
    max_area = 0

    for contour in contours:
        # Approximate the contour to reduce the number of points
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Check if the contour is rectangular and large enough
        if len(approx) == 4 and cv2.contourArea(approx) > 10000:  # Adjust area threshold
            area = cv2.contourArea(approx)
            if area > max_area:
                max_area = area
                paper_contour = approx

    # If a paper-like contour is found or previous contour exists
    if paper_contour is not None:
        previous_paper_contour = paper_contour
    elif previous_paper_contour is not None:
        # Use the previous contour if the new one is not found
        paper_contour = previous_paper_contour

    # If a paper-like contour is found
    if paper_contour is not None:
        # Draw the contour boundaries on the frame
        cv2.polylines(frame, [paper_contour], isClosed=True, color=(0, 255, 0), thickness=2)

        # Perform perspective transformation
        points = paper_contour.reshape(4, 2)

        # Sort points based on the sum of the x and y coordinates to order top-left, top-right, bottom-right, bottom-left
        points = sorted(points, key=lambda x: x[0] + x[1])  # top left smallest sum
        top_left, top_right, bottom_left, bottom_right = points

        # Define the destination points for the warp
        width, height = 640, 480  # Target dimensions for keyboard paper
        dest_points = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ], dtype="float32")

        # Compute the perspective transform matrix and apply it
        matrix = cv2.getPerspectiveTransform(np.array([top_left, top_right, bottom_right, bottom_left], dtype="float32"), dest_points)
        warped = cv2.warpPerspective(frame, matrix, (width, height))

        return frame, warped  # Return both original frame and warped paper view
    else:
        return frame, None  # Return None if no paper is found

# Initializing the drawing utils for drawing the facial landmarks on image
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hand_landmark_drawing_spec = mp_drawing.DrawingSpec(thickness=5, circle_radius=5)
hand_connection_drawing_spec = mp_drawing.DrawingSpec(thickness=10, circle_radius=10)

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

    # find paper
    image_boundaries, result = findPaperWarp(frame)

    # Display the resulting frame
    image = cv2.flip(image, 1)

    cv2.imshow("GestureFlow", image)
    # Warped paper view
    cv2.imshow("Warped Paper", result)

    # Enter key 'q' to break the loop
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# When all the process is done
# Release the capture and destroy all windows
capture.release()
cv2.destroyAllWindows()

