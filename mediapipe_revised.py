import socket
import time
import cv2
import mediapipe as mp
import numpy as np

# unity socket config

HOST = "127.0.0.1"
PORT = 5005

def send_unity(cmd: str):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((HOST, PORT))
        s.sendall(cmd.encode("utf-8"))
        s.close()
        print(f"[Python] Sent command: {cmd}")
    except ConnectionRefusedError:
        print("[Python] Unity socket not active")

# MediaPipe 

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    model_complexity=1,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

prev_wrist_x = None
prev_wrist_y = None

MOVE_THRESHOLD = 0.01
SHOT_THRESHOLD = 0.05
SHOT_COOLDOWN = 0.5 
OPEN_THRESHOLD = 1.1

LEFT_ZONE = 0.35
RIGHT_ZONE = 0.65

last_shot_time = 0

def is_hand_open(hand_landmarks):
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]

    open_fingers = 0
    for tip, pip in zip(tips, pips):
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y:
            open_fingers += 1

    return open_fingers >= 3

def hand_is_open(lm) -> bool:
    wrist = np.array([lm.landmark[0].x, lm.landmark[0].y])
    tip_idxs = [4, 8, 12, 16, 20]
    pip_idxs = [2, 6, 10, 14, 18]

    ratios = []
    for t_idx, p_idx in zip(tip_idxs, pip_idxs):
        tip = np.array([lm.landmark[t_idx].x, lm.landmark[t_idx].y])
        pip = np.array([lm.landmark[p_idx].x, lm.landmark[p_idx].y])
        pip_dist = np.linalg.norm(pip - wrist)
        if pip_dist < 1e-6:
            continue
        tip_dist = np.linalg.norm(tip - wrist)
        ratios.append(tip_dist / pip_dist)

    if not ratios:
        return False

    open_score = sum(ratios) / len(ratios)
    return open_score >= OPEN_THRESHOLD

def main():
    global prev_wrist_x, prev_wrist_y, last_shot_time

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not access webcam")
        return

    print("MediaPipe to Unity control running. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # accounting for mirrored webcame
        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            hand_landmarks = result.multi_hand_landmarks[0]

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            wrist_x = wrist.x
            wrist_y = wrist.y

            # hand_open = is_hand_open(hand_landmarks)
            hand_open = hand_is_open(hand_landmarks)
            
            if prev_wrist_x is not None:
                dx = wrist_x - prev_wrist_x
                dy = wrist_y - prev_wrist_y
                speed = abs(dx) + abs(dy)
                now = time.time()

                # CLOSED FIST
                if not hand_open:
                    if wrist_x < LEFT_ZONE:
                        send_unity("MOVE_LEFT")
                    elif wrist_x > RIGHT_ZONE:
                        send_unity("MOVE_RIGHT")
                    else:
                        send_unity("STOP")


                # OPEN HAND
                else:
                    if speed > SHOT_THRESHOLD and (now - last_shot_time) > SHOT_COOLDOWN:
                        send_unity("NORMAL_SHOT")
                        last_shot_time = now

            prev_wrist_x = wrist_x
            prev_wrist_y = wrist_y

        cv2.imshow("MediaPipe Hands to Unity", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()