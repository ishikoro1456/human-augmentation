import asyncio
import websockets
import threading
import time
import serial
import re

# ============ 設定 (main.py から移植) ============
PORT = "/dev/cu.usbserial-140" # 必要に応じて変更
BAUD = 115200
DT = 0.3
THRESH_PITCH = 20
THRESH_YAW = 25
COOLDOWN = 10.0

# WebSocketクライアントを保持するセット
connected_clients = set()
echo_active = False # エコーが現在アクティブかどうか

# =====================================
# ★ 頷き・首振り検出（reaction_detect を統合） - main.py から移植
# =====================================
def parse_line(line):
    vals = re.findall(r"[-+]?\d*\.\d+|\d+", line)
    if len(vals) >= 6:
        return tuple(map(float, vals[:6]))
    return None

async def send_echo_command(command):
    if connected_clients:
        # すべての接続済みクライアントにコマンドを送信
        await asyncio.gather(*[client.send(command) for client in connected_clients])

def head_motion_loop():
    global echo_active
    ser = serial.Serial(PORT, BAUD, timeout=1)
    print("🔍 Listening for head motion...")

    pitch_angle = 0.0
    yaw_angle = 0.0
    last_detect = 0.0

    while True:
        line = ser.readline().decode(errors='ignore').strip()
        vals = parse_line(line)
        if not vals:
            continue

        ax, ay, az, gx, gy, gz = vals

        pitch_angle += gy * DT
        yaw_angle += gz * DT
        pitch_angle *= 0.98
        yaw_angle *= 0.98

        now = time.time()

        if abs(yaw_angle) > THRESH_YAW and (now - last_detect > COOLDOWN):
            print("🟢 NOD detected! Activating echo for 3 seconds...")
            last_detect = now
            pitch_angle = yaw_angle = 0.0

            # エコー開始メッセージを送信
            asyncio.run(send_echo_command("start_echo"))
            echo_active = True

            # 3秒待機
            time.sleep(3)

            # 3秒後にエコー停止メッセージを送信
            print("🔴 Echo stopped after 3 seconds.")
            asyncio.run(send_echo_command("stop_echo"))
            echo_active = False

        # SHAKE検出ロジックは今回は不要なのでコメントアウト
        # elif abs(pitch_angle) > THRESH_PITCH and (now - last_detect > COOLDOWN):
        #     print("🔴 SHAKE detected!")
        #     last_detect = now
        #     pitch_angle = yaw_angle = 0.0
        #     # SHAKEの場合はエコーをかけない

# =====================================
# ★ WebSocketサーバー
# =====================================
async def websocket_handler(websocket, *args):
    print(f"クライアントが接続しました: {websocket.remote_address}")
    connected_clients.add(websocket)
    try:
        await websocket.wait_closed()
    finally:
        print(f"クライアントが切断しました: {websocket.remote_address}")
        connected_clients.remove(websocket)

async def start_websocket_server():
    print("WebSocketサーバーを開始します...")
    async with websockets.serve(websocket_handler, "localhost", 8765):
        await asyncio.Future()  # サーバーを永続的に実行

# =====================================
# ★ メイン実行ロジック
# =====================================
async def main():
    # 頷き検出ループを別スレッドで実行
    threading.Thread(target=head_motion_loop, daemon=True).start()

    # WebSocketサーバーをメインイベントループで実行
    await start_websocket_server()

if __name__ == "__main__":
    asyncio.run(main())
