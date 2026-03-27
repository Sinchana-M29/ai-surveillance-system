from ultralytics import YOLO
import cv2

# Global variables for drawing
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

    cv2.namedWindow("Intruder Detection")
    cv2.setMouseCallback("Intruder Detection", draw_rectangle)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)

        intruder_detected = False

        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])

                if cls == 0:  # person
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    cv2.rectangle(frame, (x1, y1), (x2, y2),
                                  (0, 255, 0), 2)

                    # Check if inside drawn zone
                    if zone:
                        zx1, zy1, zx2, zy2 = zone

                        if (x1 < zx2 and x2 > zx1 and
                                y1 < zy2 and y2 > zy1):

                            intruder_detected = True

                            cv2.putText(frame, "INTRUDER ALERT!",
                                        (50, 50),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        1.2, (0, 0, 255), 3)

        # Draw the custom zone
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
