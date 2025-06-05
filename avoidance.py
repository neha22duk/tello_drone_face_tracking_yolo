import cv2
from djitellopy import Tello
import time
import numpy as np

# Thresholds
thres = 0.45
nmsThres = 0.2

# Load class names
classNames = []
classFile = 'coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().split('\n')

# Load model
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# Initialize the Tello drone
tello = Tello()
tello.connect()
print(tello.get_battery())

# Stream setup
tello.streamoff()
tello.streamon()

# Take off and initial move
tello.takeoff()
tello.move_up(30)
tello.send_rc_control(0, 30, 0, 0)
time.sleep(2)

# Function to avoid obstacles
def avoid_obstacle(classId, box):
    x, y, w, h = box
    centerX = x + w // 2
    centerY = y + h // 2

    # Define the avoidance strategy based on the position of the detected object
    if centerX < 320 - 50:  # Object is on the left
        tello.send_rc_control(0, 0, 0, 90)  # Move right
        time.sleep(1)
    elif centerX > 320 + 50:  # Object is on the right
        tello.send_rc_control(0, 0, 0, -90)  # Move left
        time.sleep(1)
    elif centerY < 240 - 50:  # Object is above
        tello.send_rc_control(0, 90, 0, 0)  # Move down
        time.sleep(1)
    elif centerY > 240 + 50:  # Object is below
        tello.send_rc_control(0, -90, 0, 0)  # Move up
        time.sleep(1)
    else:  # Object is in front
        tello.send_rc_control(-90, 0, 0, 0)  # Move backward
        time.sleep(1)
    tello.send_rc_control(0, 0, 0, 0)  # Stop

# Main loop
try:
    while True:
        # Get the frame from the drone's camera
        img = tello.get_frame_read().frame
        classIds, confs, bbox = net.detect(img, confThreshold=thres, nmsThreshold=nmsThres)

        if len(classIds) != 0:
            for classId, conf, box in zip(classIds.flatten(), confs.flatten(), bbox):
                # Draw bounding box
                x, y, w, h = box
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # Put label
                cv2.putText(img, f'{classNames[classId - 1].upper()} {round(conf * 100, 2)}',
                            (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                # Avoid detected object
                avoid_obstacle(classId, box)
        else:
            # No objects detected, move forward
            tello.send_rc_control(0, 50, 0, 0)

        # Display the frame
        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            tello.land()
            break
finally:
    tello.end()
    cv2.destroyAllWindows()
