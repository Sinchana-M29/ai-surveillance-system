from ultralytics import YOLO
import cv2
import os
from datetime import datetime

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


def detect_intruder():
    global zone

    model = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture(0)

    # Create logs folder
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

        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])

                if cls == 0:  # person detected
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2),
                                  (0, 255, 0), 2)

                    cv2.putText(frame, "Human",
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 255, 0), 2)

                    # Check intrusion
                    if zone:
                        zx1, zy1, zx2, zy2 = zone

                        if (x1 < zx2 and x2 > zx1 and
                                y1 < zy2 and y2 > zy1):

                            # ALERT
                            cv2.putText(frame, "INTRUDER ALERT!",
                                        (50, 50),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        1.2, (0, 0, 255), 3)

                            current_time = datetime.now()

                            # Save every 5 seconds
                            if last_saved_time is None or \
                               (current_time - last_saved_time).seconds > 5:

                                timestamp = current_time.strftime(
                                    "%Y%m%d_%H%M%S")

                                filename = f"logs/intruder_{timestamp}.jpg"
                                cv2.imwrite(filename, frame)

                                with open("logs/log.txt", "a") as f:
                                    f.write(
                                        f"Intruder detected at {timestamp}\n")

                                last_saved_time = current_time

        # Draw restricted zone
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
