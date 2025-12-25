import serial

# === 設定 ===
PORT = "/dev/cu.usbserial-140"   # ここを変える
BAUD = 115200   # Arduino側と合わせる

# === 接続開始 ===
ser = serial.Serial(PORT, BAUD, timeout=1)

print(f"Connected to {PORT}. Reading data...\n")

try:
    while True:
        line = ser.readline().decode('utf-8').strip()  # 1行読み取り
        if line:
            print(line)

except KeyboardInterrupt:
    print("\n終了しました。")

finally:
    ser.close()
