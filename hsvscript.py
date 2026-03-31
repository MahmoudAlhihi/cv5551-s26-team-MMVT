import cv2, numpy
from utils.zed_camera import ZedCamera

def main():
    zed = ZedCamera()

    try:
        cv_image = zed.image

        # Convert to BGR and HSV
        bgr = cv2.cvtColor(cv_image, cv2.COLOR_BGRA2BGR)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        # Mouse callback — prints HSV at clicked pixel
        def on_mouse(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                h_val = hsv[y, x, 0]
                s_val = hsv[y, x, 1]
                v_val = hsv[y, x, 2]
                print(f'Clicked ({x}, {y}) -> HSV: ({h_val}, {s_val}, {v_val})')

        cv2.namedWindow('HSV Sampler', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('HSV Sampler', 1280, 720)
        cv2.setMouseCallback('HSV Sampler', on_mouse)
        cv2.imshow('HSV Sampler', bgr)

        print('Click on each cube face to sample HSV. Press q to quit.')
        while True:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

    finally:
        zed.close()

if __name__ == "__main__":
    main()

#lo = (min_H - margin, min_S - margin, min_V - margin)
#hi = (max_H + margin, max_S + margin, max_V + margin)