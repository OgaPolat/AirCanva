import cv2
import numpy as np

# Create videocapture object
cap = cv2.VideoCapture(0)

# Define various colors
colors = [(255, 0, 0), (255, 0, 255), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
color = colors[0]

# Get the frame dimensions
width = int(cap.get(3))
height = int(cap.get(4))

# Create a blank canvas
canvas = np.zeros((height, width, 3), np.uint8)

# Kernel for dilation
kernel = np.ones((5, 5), np.uint8)

# Previous center point for drawing
previous_center_point = None

while True:
    # Read each frame from webcam
    success, frame = cap.read()
    if not success:
        break
    
    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)

    # Adding the color buttons and clear button to the live frame
    cv2.rectangle(frame, (20, 1), (120, 65), (122, 122, 122), -1)
    cv2.rectangle(frame, (140, 1), (220, 65), colors[0], -1)
    cv2.rectangle(frame, (240, 1), (320, 65), colors[1], -1)
    cv2.rectangle(frame, (340, 1), (420, 65), colors[2], -1)
    cv2.rectangle(frame, (440, 1), (520, 65), colors[3], -1)
    cv2.rectangle(frame, (540, 1), (620, 65), colors[4], -1)
    
    cv2.putText(frame, "CLEAR ALL", (30, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "BLUE", (155, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "VIOLET", (255, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "GREEN", (355, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "RED", (465, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "YELLOW", (555, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

    # Color range for detecting green object (adjust based on object)
    lower_bound = np.array([50, 100, 100])
    upper_bound = np.array([90, 255, 255])

    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create a binary segmented mask of the green object
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    # Dilate the mask to increase the segmented area
    mask = cv2.dilate(mask, kernel, iterations=1)

    # Find contours of the segmented mask
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # contours, h    = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        # Get the biggest contour by area
        cmax = max(contours, key=cv2.contourArea)

        # Find the area of the contour
        area = cv2.contourArea(cmax)

        min_area = 1000  # Threshold area for object detection

        if area > min_area:
            # Get the center point of the contour
            M = cv2.moments(cmax)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

                # Draw a circle at the center of the contour
                cv2.circle(frame, (cX, cY), 10, (0, 0, 255), 2)

                # Check if selecting color or clear canvas
                if cY < 65:
                    if 20 < cX < 120:
                        canvas = np.zeros((height, width, 3), np.uint8)  # Clear canvas
                    elif 140 < cX < 220:
                        color = colors[0]  # Blue
                    elif 240 < cX < 320:
                        color = colors[1]  # Violet
                    elif 340 < cX < 420:
                        color = colors[2]  # Green
                    elif 440 < cX < 520:
                        color = colors[3]  # Red
                    elif 540 < cX < 620:
                        color = colors[4]  # Yellow
                else:
                    # If drawing is started, draw a line on the canvas
                    if previous_center_point is not None:
                        cv2.line(canvas, previous_center_point, (cX, cY), color, 2)

                    # Update the previous center point
                    previous_center_point = (cX, cY)
            else:
                previous_center_point = None
    else:
        previous_center_point = None

    # Merge the canvas with the frame
    canvas_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, canvas_binary = cv2.threshold(canvas_gray, 20, 255, cv2.THRESH_BINARY_INV)
    canvas_binary = cv2.cvtColor(canvas_binary, cv2.COLOR_GRAY2BGR)

    frame = cv2.bitwise_and(frame, canvas_binary)
    frame = cv2.bitwise_or(frame, canvas)

    # Display the frame with the canvas
    cv2.imshow("Frame", frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
