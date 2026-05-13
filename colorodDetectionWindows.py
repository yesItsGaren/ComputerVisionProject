import cv2
import numpy as np
import time

def find_c925e():
    for i in [1, 2, 3]:
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cap.release()
        if w == 1920:
            print(f"✅ C925e found at index {i}")
            return i
    print("❌ C925e not found — check USB connection")
    return None

# --- Cyan ball HSV values ---
lower = np.array([55, 45, 29])
upper = np.array([86, 95, 173])
# ---------------------------

MIN_AREA = 400
kernel = np.ones((5, 5), np.uint8)

index = find_c925e()
if index is None:
    exit()

cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)     # <-- only change, stops buffering old frames
cap.read()

actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Running at: {actual_w}x{actual_h}")

fps = 0.0
frame_count = 0
t_fps = time.perf_counter()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    frame_count += 1
    elapsed = time.perf_counter() - t_fps
    if elapsed >= 1.0:
        fps = frame_count / elapsed
        frame_count = 0
        t_fps = time.perf_counter()

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_AREA:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        M = cv2.moments(cnt)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.circle(frame, (cx, cy), 6, (0, 255, 255), -1)

        label = f"BALL! Area: {int(area)}  Center: ({cx},{cy})"
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    valid = [c for c in contours if cv2.contourArea(c) >= MIN_AREA]
    cv2.putText(frame, f"Objects: {len(valid)}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

    fps_color = (0, 255, 0) if fps >= 20 else (0, 165, 255) if fps >= 10 else (0, 0, 255)
    cv2.putText(frame, f"{fps:.1f} FPS", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, fps_color, 2)

    cv2.imshow("Cyan Ball Detection - C925e", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()