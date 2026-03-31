from ultralytics import YOLO
import cv2
import os
from datetime import datetime
from face_module import load_known_faces, recognize_face

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

    # Load known faces
    load_known_faces()

    # Create logs folder
    if not os.path.exists("logs"):
        os.makedirs("logs")

    last_saved_time = None

    # Face detector
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    cv2.namedWindow("Intruder Detection")
    cv2.setMouseCallback("Intruder Detection", draw_rectangle)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)

        # Detect faces
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])

                if cls == 0:  # person
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    cv2.rectangle(frame, (x1, y1), (x2, y2),
                                  (0, 255, 0), 2)

                    cv2.putText(frame, "Human",
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 255, 0), 2)

        # Face recognition logic
        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            name = recognize_face(face_img)

            if name == "Unknown":
                cv2.putText(frame, "UNKNOWN",
                            (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (0, 0, 255), 2)

                # Check zone intrusion
                if zone:
                    zx1, zy1, zx2, zy2 = zone

                    if (x < zx2 and x+w > zx1 and
                            y < zy2 and y+h > zy1):

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
                                f.write(f"Intruder detected at {timestamp}\n")

                            last_saved_time = current_time

            else:
                cv2.putText(frame, f"Authorized: {name}",
                            (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 0), 2)

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
