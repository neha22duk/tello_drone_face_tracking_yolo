from djitellopy import tello
import cv2

me = tello.Tello()
me.connect()
print(me.get_battery())

me.streamon()

while True:
    img = me.get_frame_read().frame
    img = cv2.resize(img, (360,240))
    cv2.imshow('image',img)
    cv2.waitKey(1)

