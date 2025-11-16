import cv2
import mediapipe as mp

def main():
    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        model_complexity=1,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    mp_draw = mp.solutions.drawing_utils
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not access webcam")
        return
    print("MediaPipe Hands running... Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Process frame
        result = hands.process(rgb)
        # Draw hand landmarks
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2)
                )
        # Show frame
        cv2.imshow("MediaPipe Hands (Press q to quit)", frame)
        # Exit on q
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
