from unity_sender import send_unity
import time

send_unity("MOVE_LEFT")
time.sleep(1)

send_unity("MOVE_RIGHT")
time.sleep(1)

send_unity("NORMAL_SHOT")
