import cv2
from djitellopy import tello


thres = 0.55
nmsThres = 0.2
# cap = cv2.VideoCapture(0)
# cap.set(3, 640)
# cap.set(4, 480)

classNames = []
classFile = 'coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().split('\n')
print(classNames)
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = "frozen_inference_graph.pb"

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

me = tello.Tello()
me.connect()
print(me.get_battery())
me.streamoff()
me.streamon()

me.takeoff()
me.move_up(80)

while True:
    # success, img = cap.read()
    img = me.get_frame_read().frame
    classIds, confs, bbox = net.detect(img, confThreshold=thres, nmsThreshold=nmsThres)
    try:
        for classId, conf, box in zip(classIds.flatten(), confs.flatten(), bbox):
            # Draw rectangle
            cv2.rectangle(img, box, (0, 255, 0), 2)
            # Put text
            cv2.putText(img, f'{classNames[classId - 1].upper()} {round(conf * 100, 2)}',
                        (box[0], box[1] - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                        1, (0, 255, 0), 2)
    except Exception as e:
        print(f"Error: {e}")
        pass

    me.send_rc_control(0, 0, 0, 0)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
def avoid_obstacle(classId, box):
    x, y, w, h = box
    centerX = x + w // 2

    centerY = y + h // 2

    print(f"Object detected: classId={classId}, centerX={centerX}, centerY={centerY}")

    # Define the avoidance strategy based on the position of the detected object
    if centerX < 320 - 50:  # Object is on the left
        print("Moving right")
        me.send_rc_control(0, 0, 0, 30)  # Move right
    elif centerX > 320 + 50:  # Object is on the right
        print("Moving left")
        me.send_rc_control(0, 0, 0, -30)  # Move left
    elif centerY < 240 - 50:  # Object is above
        print("Moving down")
        me.send_rc_control(0, 30, 0, 0)  # Move down
    elif centerY > 240 + 50:  # Object is below
        print("Moving up")
        me.send_rc_control(0, -30, 0, 0)  # Move up
    else:  # Object is in front
        print("Moving backward")
        me.send_rc_control(-30, 0, 0, 0)  # Move backward

# Main loop
while True:
    # Get the frame from the drone's camera
    img = me.get_frame_read().frame
    classIds, confs, bbox = net.detect(img, confThreshold=thres, nmsThreshold=nmsThres)

    if len(classIds) != 0:
        for classId, conf, box in zip(classIds.flatten(), confs.flatten(), bbox):
            # Draw bounding box and label
            cvzone.cornerRect(img, box)
            cv2.putText(img, f'{classNames[classId - 1].upper()} {round(conf * 100, 2)}',
                        (box[0] + 10, box[1] + 30), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                        1, (0, 255, 0), 2)
            # Avoid detected object
            avoid_obstacle(classId, box)
    else:
        # No objects detected, hover in place
        print("Hovering in place")
        me.send_rc_control(0, 0, 0, 0)

    # Display the frame
    cv2.imshow("Image", img)
    cv2.waitKey(1)
