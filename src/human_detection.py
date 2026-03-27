from ultralytics import YOLO
import cv2


def detect_humans():
    # Load YOLO model
    model = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run detection
        results = model(frame)

        # Draw results
        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])

                # Class 0 = person
                if cls == 0:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    cv2.rectangle(frame, (x1, y1), (x2, y2),
                                  (0, 255, 0), 2)

                    cv2.putText(frame, "Human Detected",
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.9, (0, 0, 255), 2)

        cv2.imshow("Human Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
