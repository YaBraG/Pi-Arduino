import cv2
import numpy as np
import serial
import time

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Global variables
color_to_track = None
lower_color_bound = None
upper_color_bound = None
tracking_enabled = False
floatPserv = 90
floatTserv = 90
intPserv=90
intTserv=90
strPserv=""
strTserv=""
pan_x=""
tilt_y=""
try:
    arduino = serial.Serial('/dev/ttyACM1', 9600, timeout=1)
except:
    arduino = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
arduino.reset_input_buffer()

# Function to handle mouse clicks


def select_color(event, x, y, flags, param):
    global color_to_track, lower_color_bound, upper_color_bound, tracking_enabled

    if event == cv2.EVENT_LBUTTONDOWN:
        # Capture the color of the pixel clicked
        color_to_track = frame[y, x, :].copy()

        # Convert to HSV
        hsv_color = cv2.cvtColor(
            np.uint8([[color_to_track]]), cv2.COLOR_BGR2HSV)[0][0]

        # Define a range around the selected color for tracking
        hue_sensitivity = 10
        sv_sensitivity = 40
        lower_color_bound = np.array([max(hsv_color[0] - hue_sensitivity, 0), max(
            hsv_color[1] - sv_sensitivity, 0), max(hsv_color[2] - sv_sensitivity, 0)])
        upper_color_bound = np.array([min(hsv_color[0] + hue_sensitivity, 180), min(
            hsv_color[1] + sv_sensitivity, 255), min(hsv_color[2] + sv_sensitivity, 255)])

        tracking_enabled = True


cv2.namedWindow("Color Tracking with Click")
cv2.setMouseCallback("Color Tracking with Click", select_color)

targetsquare = 100

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    frame_height, frame_width = frame.shape[:2]
    center_x, center_y = frame_width // 2, frame_height // 2

    # Draw a small square at the center of the frame
    cv2.rectangle(frame, (center_x - 5, center_y - 5),
                  (center_x + 5, center_y + 5), (0, 255, 0), -1)

    # Draw the firing area square based on the firing_area_size
    firing_area_start = (center_x - targetsquare, center_y - targetsquare)
    firing_area_end = (center_x + targetsquare, center_y + targetsquare)
    cv2.rectangle(frame, firing_area_start, firing_area_end, (255, 0, 0), 2)

    if tracking_enabled and color_to_track is not None:
        # Convert the frame to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Create a mask with the specified color range
        mask = cv2.inRange(hsv, lower_color_bound, upper_color_bound)

        # Find contours in the mask
        contours, _ = cv2.findContours(
            mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Find the largest contour and its centroid
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

                # Draw the centroid
                cv2.circle(frame, (cX, cY), 5, (255, 0, 0), -1)

                # Calculate the relative (x,y) coordinates

                relative_x, relative_y = cX - center_x, cY - center_y

                # Check if the target is within the specified range
                if -targetsquare <= relative_x <= targetsquare and -targetsquare <= relative_y <= targetsquare:
                    cv2.putText(frame, "FIRE", (center_x - targetsquare - 25, center_y -
                                targetsquare - 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                else:
                    if relative_x < 0:
                        pan_x = "pan left"
                        floatPserv = floatPserv+0.01
                    elif relative_x > 0:
                        pan_x = "pan right"
                        floatPserv = floatPserv-0.01
                    if relative_y < 0:
                        tilt_y = "tilt up"
                        floatTserv = floatTserv+0.01
                    elif relative_y > 0:
                        tilt_y = "tilt down"
                        floatTserv = floatTserv-0.01
                    if floatPserv > 180:
                        floatPserv = 180
                    if floatPserv < 0:
                        floatPserv = 0
                    if floatTserv > 180:
                        floatTserv = 180
                    if floatTserv < 90:
                        floatTserv = 90
                    intPserv = int(floatPserv)
                    intTserv = int(floatTserv)
                    strPserv = str(intPserv)
                    strTserv = str(strTserv)
                    arduino.write((strPserv+";"+strTserv+"\n").encode('utf-8'))

                print("Relative Position: (", relative_x, " , ", relative_y,
                      ") , (", pan_x, ",", tilt_y, ") (", floatPserv, " , ", floatTserv, ')')
    # Display the resulting frame
    cv2.imshow('Color Tracking with Click', frame)

    # Break the loop with 'n' or 'q'
    key = cv2.waitKey(1) & 0xFF
    if key == ord('n'):
        tracking_enabled = False
    elif key == ord('q'):
        break

# Release the capture and close any open windows
cap.release()
cv2.destroyAllWindows()
