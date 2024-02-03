import time
import cv2
import numpy as np
from AX12 import Ax12
from simple_pid import PID


def remap(x, oMin, oMax, nMin, nMax):

    # range check
    if oMin == oMax:
        print("Warning: Zero input range")
        return None

    if nMin == nMax:
        print("Warning: Zero output range")
        return None

    # check reversed input range
    reverseInput = False
    oldMin = min(oMin, oMax)
    oldMax = max(oMin, oMax)
    if not oldMin == oMin:
        reverseInput = True

    # check reversed output range
    reverseOutput = False
    newMin = min(nMin, nMax)
    newMax = max(nMin, nMax)
    if not newMin == nMin:
        reverseOutput = True

    portion = (x-oldMin)*(newMax-newMin)/(oldMax-oldMin)
    if reverseInput:
        portion = (oldMax-x)*(newMax-newMin)/(oldMax-oldMin)

    result = portion + newMin
    if reverseOutput:
        result = newMax - portion

    return result


Ax12.DEVICENAME = '/dev/ttyUSB0'
Ax12.BAUDRATE = 1000000

Ax12.connect()

motor1 = Ax12(1)
motor2 = Ax12(2)

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Global variables
color_to_track = None
lower_color_bound = None
upper_color_bound = None
tracking_enabled = False
floatPserv = 512
floatTserv = 383
intPserv = 512
intTserv = 383
pan_x = ""
tilt_y = ""
addition = 3.5
dinamixLim = 1023
targetsquare = 25
Px = 1
Dx = 0
Ix = 0
i = 0
previous_errorX = 0
motor1.set_goal_position(512)
motor2.set_goal_position(383)


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
        hue_sensitivity = 1
        sv_sensitivity = 10
        lower_color_bound = np.array([max(hsv_color[0] - hue_sensitivity, 0), max(
            hsv_color[1] - sv_sensitivity, 0), max(hsv_color[2] - sv_sensitivity, 0)])
        upper_color_bound = np.array([min(hsv_color[0] + hue_sensitivity, 180), min(
            hsv_color[1] + sv_sensitivity, 255), min(hsv_color[2] + sv_sensitivity, 255)])

        tracking_enabled = True


cv2.namedWindow("Color Tracking with Click")
cv2.setMouseCallback("Color Tracking with Click", select_color)


while True:
    # Capture frame-by-frame
    previous = time.time()
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
                    errorX = cX - center_x
                    errorY = cY - center_y
                    p = Px * errorX
                    d = Dx * ((errorX - previous_errorX) / elapsedtime)
                    if -targetsquare+5 < errorX < targetsquare+5:
                        i = i + (Ix * errorX)
                    pid = p + i + d
                    # if relative_x < 0:
                    #     pan_x = "pan left"
                    #     floatPserv = floatPserv+addition
                    # elif relative_x > 0:
                    #     pan_x = "pan right"
                    #     floatPserv = floatPserv-addition
                    # if relative_y < 0:
                    #     tilt_y = "tilt down"
                    #     floatTserv = floatTserv-addition
                    # elif relative_y > 0:
                    #     tilt_y = "tilt up"
                    #     floatTserv = floatTserv+addition
                    # if floatPserv > dinamixLim:
                    #     floatPserv = dinamixLim
                    # if floatPserv < 0:
                    #     floatPserv = 0
                    # if floatTserv > dinamixLim:
                    #     floatTserv = dinamixLim
                    # if floatTserv < 0:
                    #     floatTserv = 0
                    # floatPserv = round(floatPserv, 2)
                    # floatTserv = round(floatTserv, 2)
                    # intPserv = int(floatPserv)
                    # intTserv = int(floatTserv)
                    # motor1.set_goal_position(intPserv)
                    # motor2.set_goal_position(intTserv)
                    previous_errorX = errorX
                    print(pid)

                # print("Relative Position: (", relative_x, " , ", relative_y,
                #       ") , (", frame_height, ",", frame_width, ") (", floatPserv, " , ", floatTserv, ')')
                # print("| FPan: ", floatPserv, " | FTilt: ", floatTserv, " | IPan: ", intPserv,
                #       " | ITilt: ", intTserv)
    # Display the resulting frame
    cv2.imshow('Color Tracking with Click', frame)
    current = time.time()
    elapsedtime = current - previous

    # Break the loop with 'n' or 'q'
    key = cv2.waitKey(1) & 0xFF
    if key == ord('n'):
        tracking_enabled = False
        intPserv = 512
        intTserv = 383
        floatPserv = 512
        floatTserv = 383
        motor1.set_goal_position(512)
        motor2.set_goal_position(383)
    elif key == ord('q'):
        motor1.set_goal_position(512)
        motor2.set_goal_position(383)
        break

# Release the capture and close any open windows
cap.release()
cv2.destroyAllWindows()
motor1.set_torque_enable(0)
motor2.set_torque_enable(0)
Ax12.disconnect()
