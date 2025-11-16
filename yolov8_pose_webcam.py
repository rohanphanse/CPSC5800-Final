import cv2
from ultralytics import YOLO

def main():
    # Load YOLO pose model
    model = YOLO("yolov8n-pose.pt")

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not access webcam")
        return

    print("YOLOv8 Pose running... Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break

        results = model(frame)
        r = results[0]
        annotated_frame = r.plot()

        cv2.imshow("YOLOv8 Pose (Press q to quit)", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
