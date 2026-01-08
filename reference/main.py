import threading
import time
import serial
import re

import socketio

# ============ IMU / 検出設定 ============
PORT = "/dev/cu.usbserial-140"
BAUD = 115200
DT = 0.3
THRESH_PITCH = 20
THRESH_YAW = 25
COOLDOWN = 10.0
# =====================================

# ============ ネットワーク設定 ============
SERVER_URL = "http://192.168.68.63:3000"  # ←中継サーバのIPに変更
ROOM_ID = "room1"
USER_ID = "listener1"                    # ←聞き手識別子（任意）
# =======================================

sio = socketio.Client()

@sio.event
def connect():
    print("✅ connected to relay server")
    sio.emit("join", {"roomId": ROOM_ID, "role": "listener", "userId": USER_ID})

@sio.event
def disconnect():
    print("❌ disconnected")

def emit_reaction(kind: str, strength: float = 1.0):
    """
    kind: "nod" or "shake"
    strength: 0.0 - 1.0 (今は固定でOK)
    """
    payload = {
        "from": USER_ID,
        "type": kind,
        "strength": float(strength),
        "ts": int(time.time() * 1000),
    }
    sio.emit("reaction", payload)
    print(f"📡 sent: {payload}")

def parse_line(line: str):
    vals = re.findall(r"[-+]?\d*\.\d+|\d+", line)
    if len(vals) >= 6:
        return tuple(map(float, vals[:6]))
    return None

def head_motion_loop():
    ser = serial.Serial(PORT, BAUD, timeout=1)
    print("🔍 Listening for head motion...")

    pitch_angle = 0.0
    yaw_angle = 0.0
    last_detect = 0.0

    while True:
        line = ser.readline().decode(errors="ignore").strip()
        vals = parse_line(line)
        if not vals:
            continue

        ax, ay, az, gx, gy, gz = vals

        # 元コードの積分・減衰を踏襲
        pitch_angle += gy * DT
        yaw_angle += gz * DT
        pitch_angle *= 0.98
        yaw_angle *= 0.98

        now = time.time()

        # ※あなたの元コードの判定ロジックをそのまま使用
        if abs(pitch_angle) > THRESH_PITCH and (now - last_detect > COOLDOWN):
            print("🔴 SHAKE detected!")
            last_detect = now
            emit_reaction("shake", 1.0)
            pitch_angle = yaw_angle = 0.0

        elif abs(yaw_angle) > THRESH_YAW and (now - last_detect > COOLDOWN):
            print("🟢 NOD detected!")
            last_detect = now
            emit_reaction("nod", 1.0)
            pitch_angle = yaw_angle = 0.0

def main():
    print("🌐 connecting to relay server...")
    sio.connect(SERVER_URL, transports=["websocket"])  # LANならこれでOK

    threading.Thread(target=head_motion_loop, daemon=True).start()

    print("\n🎧 頷き / 首振りを検出すると、話し手PCへ送信します\n")
    while True:
        time.sleep(1)

if __name__ == "__main__":
    main()