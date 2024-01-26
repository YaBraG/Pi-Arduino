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
pserv = 90
tserv = 90

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

targetsquare = 50

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
                pserv = int(pserv)
                tserv = int(tserv)
                if relative_x < 0:
                    pan_x = "pan left"
                    pserv = pserv-1
                else:
                    pan_x = "pan right"
                    pserv = pserv+1
                if relative_y < 0:
                    tilt_y = "tilt up"
                    tserv = tserv+1
                else:
                    tilt_y = "tilt down"
                    tserv = tserv-1
                if pserv > 180:
                    pserv = 180
                if pserv < 0:
                    pserv = 0
                if tserv > 180:
                    tserv = 180
                if tserv < 90:
                    tserv = 90

                print("Relative Position: (", relative_x, " , ", relative_y,
                      ") , (", pan_x, ",", tilt_y, ") (", pserv, " , ", tserv, ')')
                pserv = str(pserv)
                tserv = str(tserv)

                arduino.write((pserv+";"+tserv+"\n").encode('utf-8'))
                time.sleep(0.60)

                # Check if the target is within the specified range
                if -targetsquare <= relative_x <= targetsquare and -targetsquare <= relative_y <= targetsquare:
                    cv2.putText(frame, "FIRE", (center_x - targetsquare - 25, center_y -
                                targetsquare - 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

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
