import cv2
from djitellopy import tello
from ultralytics import YOLO

# Set confidence threshold
thres = 0.55
nmsThres = 0.2

# Initialize Tello drone
me = tello.Tello()
me.connect()
print(f"Battery: {me.get_battery()}%")
me.streamoff()
me.streamon()

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # yolov8n.pt is a small model. Use other versions like yolov8s.pt, yolov8m.pt, etc., for more accuracy.

# Take off and move up
#me.takeoff()
#me.move_up(80)

while True:
    # Get frame from drone
    img = me.get_frame_read().frame

    # Perform object detection using YOLOv8
    results = model(img)

    # Draw bounding boxes and labels
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            cls = int(box.cls[0])
            label = model.names[cls]

            if conf > thres:
                # Draw rectangle
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Put text
                cv2.putText(img, f'{label.upper()} {round(conf * 100, 2)}%', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show the image
    cv2.imshow("Image", img)

    # Send zero control commands to keep the drone stable
    me.send_rc_control(0, 0, 0, 0)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Land the drone
me.land()
cv2.destroyAllWindows()
