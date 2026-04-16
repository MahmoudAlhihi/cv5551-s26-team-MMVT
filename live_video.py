import cv2
from utils.zed_camera import ZedCamera

zed = ZedCamera()

cv2.namedWindow('ZED 2 - Live', cv2.WINDOW_NORMAL)
cv2.resizeWindow('ZED 2 - Live', 1280, 720)

try:
    while True:
        frame = zed.image
        cv2.imshow('ZED 2 - Live', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    zed.close()