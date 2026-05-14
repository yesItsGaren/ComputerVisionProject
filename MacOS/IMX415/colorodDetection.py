import cv2
import numpy as np
import time

def find_imx415():
    for i in [0, 1, 2]:
        cap = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION)
        if not cap.isOpened():
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cap.release()
        if w == 3840:
            return i
    return None

# --- Paste your calibrated values here ---
lower = np.array([5, 212, 105])
upper = np.array([12, 255, 255])
# -----------------------------------------

MIN_AREA = 400  # ignore tiny blobs (noise)

cap = cv2.VideoCapture(find_imx415(), cv2.CAP_AVFOUNDATION)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)

frame_times = []

while True:
    t_start = time.perf_counter()
    ret, frame = cap.read()
    if not ret:
        break

    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)

    # Clean up mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)  # remove noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # fill holes

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_AREA:
            continue

        # Bounding box
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Centroid
        M = cv2.moments(cnt)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)

        # Label
        label = f"MUG! Area: {int(area)}  Center: ({cx},{cy})"
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 0), 2)

    # Object count
    valid = [c for c in contours if cv2.contourArea(c) >= MIN_AREA]
    cv2.putText(frame, f"Objects: {len(valid)}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

    # --- FPS ---
    frame_times.append(time.perf_counter() - t_start)
    if len(frame_times) > 30:
        frame_times.pop(0)
    fps = 1.0 / (sum(frame_times) / len(frame_times))
    fps_color = (0, 255, 0) if fps >= 20 else (0, 165, 255) if fps >= 10 else (0, 0, 255)
    cv2.putText(frame, f"{fps:.1f} FPS", (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, fps_color, 3)
    # -----------# --- FPS ---
    frame_times.append(time.perf_counter() - t_start)
    if len(frame_times) > 30:
        frame_times.pop(0)
    fps = 1.0 / (sum(frame_times) / len(frame_times))
    fps_color = (0, 255, 0) if fps >= 20 else (0, 165, 255) if fps >= 10 else (0, 0, 255)
    cv2.putText(frame, f"{fps:.1f} FPS", (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, fps_color, 3)
    # -----------

    display = cv2.resize(frame, (1920, 1080))
    cv2.imshow("Color Object Detection", display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()