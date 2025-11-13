import serial, time, re, random
import pygame

PORT = "/dev/cu.usbserial-1140"
BAUD = 115200
DT = 0.3
THRESH_PITCH = 10
THRESH_YAW = 15
COOLDOWN = 2.0

# --- 音声ファイルのパス ---
NOD_SOUNDS = [
    "sound/backchannel_affirmation_1_うんうん.mp3",
    "sound/backchannel_affirmation_2_そうそう.mp3",
    "sound/backchannel_affirmation_3_うんそうだね.mp3",
    "sound/backchannel_affirmation_4_なるほどね.mp3",
    "sound/backchannel_affirmation_5_それわかる.mp3"
]
SHAKE_SOUNDS = [
    "sound/backchannel_negation_1_いやいや.mp3",
    "sound/backchannel_negation_2_ううん.mp3",
    "sound/backchannel_negation_3_ちがうかな.mp3",
    "sound/backchannel_negation_4_うーんどうだろう.mp3",
    "sound/backchannel_negation_5_えー.mp3"
]

pygame.mixer.init()

ser = serial.Serial(PORT, BAUD, timeout=1)
print("Listening for nod/shake...")

pitch_angle = 0.0
yaw_angle = 0.0
last_detect = 0.0

def parse_line(line):
    vals = re.findall(r"[-+]?\d*\.\d+|\d+", line)
    if len(vals) >= 6:
        return tuple(map(float, vals[:6]))
    return None

def play_sound(path):
    try:
        pygame.mixer.music.load(path)
        pygame.mixer.music.play()
    except Exception as e:
        print(f"[Error] Failed to play {path}: {e}")

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

    print(f"pitch={pitch_angle:.2f}, yaw={yaw_angle:.2f}")

    now = time.time()

    # --- 首振り（No）---
    if abs(pitch_angle) > THRESH_PITCH and (now - last_detect > COOLDOWN):
        print("🔴 SHAKE detected!")
        play_sound(random.choice(SHAKE_SOUNDS))
        pitch_angle = yaw_angle = 0.0
        last_detect = now

    # --- 頷き（Yes）---
    elif abs(yaw_angle) > THRESH_YAW and (now - last_detect > COOLDOWN):
        print("🟢 NOD detected!")
        play_sound(random.choice(NOD_SOUNDS))
        pitch_angle = yaw_angle = 0.0
        last_detect = now
