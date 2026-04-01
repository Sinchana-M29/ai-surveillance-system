from ultralytics import YOLO
import cv2
import os
from datetime import datetime
import winsound
import threading
from deep_sort_realtime.deepsort_tracker import DeepSort

# Mouse drawing variables
drawing = False
ix, iy = -1, -1
zone = None


def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, zone

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            zone = (ix, iy, x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        zone = (ix, iy, x, y)


# 🔊 Alarm
def play_alarm():
    winsound.Beep(1200, 700)


def detect_intruder():
    global zone

    model = YOLO("yolov8n.pt")
    tracker = DeepSort(max_age=30)

    cap = cv2.VideoCapture(0)

    if not os.path.exists("logs"):
        os.makedirs("logs")

    last_saved_time = None

    cv2.namedWindow("Intruder Detection")
    cv2.setMouseCallback("Intruder Detection", draw_rectangle)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)

        detections = []

        # Collect detections
        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                if cls == 0:  # person
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    w, h = x2 - x1, y2 - y1

                    detections.append(([x1, y1, w, h], conf, 'person'))

        # Track objects
        tracks = tracker.update_tracks(detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            l, t, w, h = map(int, track.to_ltrb())

            cv2.rectangle(frame, (l, t), (l+w, t+h), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {track_id}",
                        (l, t - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 0), 2)

            # Intrusion check
            if zone:
                zx1, zy1, zx2, zy2 = zone

                if (l < zx2 and l+w > zx1 and
                        t < zy2 and t+h > zy1):

                    cv2.putText(frame, "INTRUDER ALERT!",
                                (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1.2, (0, 0, 255), 3)

                    current_time = datetime.now()

                    if last_saved_time is None or \
                       (current_time - last_saved_time).seconds > 5:

                        timestamp = current_time.strftime("%Y%m%d_%H%M%S")

                        filename = f"logs/intruder_{timestamp}.jpg"
                        cv2.imwrite(filename, frame)

                        with open("logs/log.txt", "a") as f:
                            f.write(f"Intruder ID {track_id} at {timestamp}\n")

                        threading.Thread(target=play_alarm,
                                         daemon=True).start()

                        last_saved_time = current_time

        # Draw zone
        if zone:
            zx1, zy1, zx2, zy2 = zone
            cv2.rectangle(frame, (zx1, zy1), (zx2, zy2),
                          (255, 0, 0), 2)

            cv2.putText(frame, "Restricted Area",
                        (zx1, zy1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 0, 0), 2)

        cv2.imshow("Intruder Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
