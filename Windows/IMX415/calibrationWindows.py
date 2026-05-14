import cv2
import numpy as np

def find_imx415():
    for i in [0, 1, 2, 3]:
        cap = cv2.VideoCapture(i, cv2.CAP_MSMF)
        if not cap.isOpened():
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_FOURCC,       cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  3840)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cap.release()
        if w == 3840:
            print(f"IMX415 found at index {i}")
            return i
    print("IMX415 not found — check USB connection")
    return None

def nothing(x):
    pass

index = find_imx415()
if index is None:
    exit()

cap = cv2.VideoCapture(index, cv2.CAP_MSMF)
cap.set(cv2.CAP_PROP_FOURCC,       cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_FPS,          60)

# grab one frame first before creating windows — fixes Windows timing issue
cap.read()

cv2.namedWindow("Trackbars")
cv2.namedWindow("Original | Mask | Result")

cv2.createTrackbar("H Min", "Trackbars", 0,   179, nothing)
cv2.createTrackbar("H Max", "Trackbars", 179, 179, nothing)
cv2.createTrackbar("S Min", "Trackbars", 0,   255, nothing)
cv2.createTrackbar("S Max", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("V Min", "Trackbars", 0,   255, nothing)
cv2.createTrackbar("V Max", "Trackbars", 255, 255, nothing)

print("Calibration running — adjust trackbars until your object is white in the mask")
print("Press S to save values, Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame grab failed")
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    h_min = cv2.getTrackbarPos("H Min", "Trackbars")
    h_max = cv2.getTrackbarPos("H Max", "Trackbars")
    s_min = cv2.getTrackbarPos("S Min", "Trackbars")
    s_max = cv2.getTrackbarPos("S Max", "Trackbars")
    v_min = cv2.getTrackbarPos("V Min", "Trackbars")
    v_max = cv2.getTrackbarPos("V Max", "Trackbars")

    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    mask  = cv2.inRange(hsv, lower, upper)

    result = cv2.bitwise_and(frame, frame, mask=mask)

    display  = cv2.resize(frame,  (640, 360))
    mask_d   = cv2.resize(mask,   (640, 360))
    result_d = cv2.resize(result, (640, 360))

    mask_bgr = cv2.cvtColor(mask_d, cv2.COLOR_GRAY2BGR)
    combined = np.hstack([display, mask_bgr, result_d])

    cv2.imshow("Original | Mask | Result", combined)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        print(f"\nYour HSV values:")
        print(f"lower = np.array([{h_min}, {s_min}, {v_min}])")
        print(f"upper = np.array([{h_max}, {s_max}, {v_max}])")

cap.release()
cv2.destroyAllWindows()
