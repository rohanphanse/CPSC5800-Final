import socket

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
        print("Socket not active.")
  
