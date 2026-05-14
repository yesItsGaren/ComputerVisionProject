import cv2
import numpy as np

def nothing(x):
    pass

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# grab one frame first before creating windows — fixes Windows timing issue
cap.read()

cv2.namedWindow("Trackbars")
cv2.namedWindow("Original | Mask | Result")

cv2.createTrackbar("H Min", "Trackbars", 80,  179, nothing)
cv2.createTrackbar("H Max", "Trackbars", 100, 179, nothing)
cv2.createTrackbar("S Min", "Trackbars", 100, 255, nothing)
cv2.createTrackbar("S Max", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("V Min", "Trackbars", 50,  255, nothing)
cv2.createTrackbar("V Max", "Trackbars", 255, 255, nothing)

print("Calibration running — adjust trackbars until your cyan ball is white in the mask")
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
        print(f"\n✅ Your CYAN HSV values:")
        print(f"lower = np.array([{h_min}, {s_min}, {v_min}])")
        print(f"upper = np.array([{h_max}, {s_max}, {v_max}])")

cap.release()
cv2.destroyAllWindows()