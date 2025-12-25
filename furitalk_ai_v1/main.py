import threading
import queue
import time
import wave
import subprocess
import pyaudio
import openai
import serial
import re

# ============ 設定 ============
PORT = "/dev/cu.usbserial-140"
BAUD = 115200
DT = 0.3
THRESH_PITCH = 20
THRESH_YAW = 25
COOLDOWN = 10.0

BUFFER_SECONDS = 8   # ★ 直近8秒を保持
RATE = 16000
CHUNK = 1024
MIN_SEC = 0.5

client = openai.OpenAI()

is_playing_audio = False   # 再生中かどうか
is_processing_reaction = False  # 相槌処理中かどうか
# =====================================


# =====================================
# ★ 1. オーディオ録音スレッド（常時録音 → 循環バッファへ）
# =====================================
audio_q = queue.Queue(maxsize=int(BUFFER_SECONDS * RATE / CHUNK))

def audio_record_loop():
    global is_playing_audio

    pa = pyaudio.PyAudio()
    stream = pa.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK
    )

    print("🎤 Audio recording started...")

    while True:
        data = stream.read(CHUNK)

        # ★ 再生中は録音はするがバッファには詰めない
        if is_playing_audio:
            continue

        if audio_q.full():
            audio_q.get()  # 古いデータを捨てる
        audio_q.put(data)


# 録音スレッド起動
threading.Thread(target=audio_record_loop, daemon=True).start()


# =====================================
# ★ 2. バッファ内容を WAV として保存
# =====================================
def save_buffer_to_wav(path="input.wav"):
    frames = list(audio_q.queue)  # 現在のバッファをコピー
    # 0.5秒分以上のチャンクがなければ諦める
    if len(frames) * CHUNK < RATE * MIN_SEC:
        return False

    # ★ ここで空チェック
    if not frames:   # len(frames) == 0 と同じ
        return False

    wf = wave.open(path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    return True



# =====================================
# ★ 3. Whisper で文字起こし
# =====================================
def speech_to_text(filename="input.wav"):
    with open(filename, "rb") as f:
        res = client.audio.transcriptions.create(
            model="gpt-4o-transcribe",
            file=f
        )
    return res.text


# =====================================
# ★ 4. 相槌生成（肯定 / 否定でプロンプト分岐）
# =====================================
def generate_backchannel(user_text, motion_type):
    """
    motion_type:
        "nod"   -> 肯定の相槌
        "shake" -> 否定の相槌
    """

    if motion_type == "nod":
        # 頷き：肯定・共感系の相槌
        prompt = f"""あなたは会話相槌AIです。
ユーザの発話内容を理解したうえで、肯定・共感・同意を示す自然な短い日本語の相槌を返してください。

- 返答は二文以内、20〜30文字ぐらいまでにしてください。
- 敬語 / ですます調を基本としてください。
- ユーザの発話内容を踏まえた返答をしてください。

ユーザ: {user_text}
相槌："""
    elif motion_type == "shake":
        # 首振り：否定・違いを穏やかに受け止める相槌
        prompt = f"""あなたは会話相槌AIです。
ユーザの発話内容を理解したうえで、「否定」や「そうではない」というニュアンスを穏やかに受け止める自然な短い日本語の相槌を返してください。

- 相手を責めたり強く否定したりする表現は避けてください。
- 返答は二文以内、20〜30文字ぐらいまでにしてください。
- 敬語 / ですます調を基本としてください。
- ユーザの発話内容を踏まえた返答をしてください。

ユーザ: {user_text}
相槌："""
    else:
        # 保険：従来と同じ無難な相槌
        prompt = f"ユーザの発話に自然で短い相槌を返してください。\nユーザ:{user_text}\n相槌:"

    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return res.choices[0].message.content


# =====================================
# ★ 5. 音声生成（TTS）
# =====================================
def text_to_wav(text, out_path="reply.wav"):
    res = client.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice="alloy",
        input=text,
        response_format="wav"
    )
    data = res.read()
    with open(out_path, "wb") as f:
        f.write(data)


# =====================================
# ★ 6. 音声再生（afplay）
# =====================================
def clear_audio_buffer():
    while not audio_q.empty():
        try:
            audio_q.get_nowait()
        except queue.Empty:
            break

def play_audio(path="reply.wav"):
    global is_playing_audio
    is_playing_audio = True
    try:
        subprocess.run(["afplay", path])
    finally:
        is_playing_audio = False
        clear_audio_buffer()



# =====================================
# ★ 7. 頷き・首振り検出（reaction_detect を統合）
# =====================================
def parse_line(line):
    vals = re.findall(r"[-+]?\d*\.\d+|\d+", line)
    if len(vals) >= 6:
        return tuple(map(float, vals[:6]))
    return None


def head_motion_loop():
    global is_processing_reaction

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

        # ★ 処理中は積分をリセットしてスキップ
        if is_processing_reaction:
            pitch_angle = yaw_angle = 0.0
            continue

        ax, ay, az, gx, gy, gz = vals

        pitch_angle += gy * DT
        yaw_angle += gz * DT
        pitch_angle *= 0.98
        yaw_angle *= 0.98

        now = time.time()

        if abs(pitch_angle) > THRESH_PITCH and (now - last_detect > COOLDOWN):
            print("🔴 SHAKE detected!")
            last_detect = now
            handle_reaction("shake")
            pitch_angle = yaw_angle = 0.0

        elif abs(yaw_angle) > THRESH_YAW and (now - last_detect > COOLDOWN):
            print("🟢 NOD detected!")
            last_detect = now
            handle_reaction("nod")
            pitch_angle = yaw_angle = 0.0



# =====================================
# ★ 8. 反応処理（録音→テキスト→相槌→音声→再生）
# =====================================
def handle_reaction(motion_type):
    global is_processing_reaction

    if is_processing_reaction:
        print("⏳ Reaction already in progress. Ignored.")
        return

    is_processing_reaction = True
    try:
        print("💾 Saving audio buffer...")
        ok = save_buffer_to_wav("input.wav")
        if not ok:
            print("⚠️ Audio buffer is empty. Skipping transcription.")
            return

        print("📝 Transcribing...")
        text = speech_to_text("input.wav")
        print(f"You said: {text}")

        if len(text.strip()) == 0:
            print("⚠️ No speech detected.")
            return

        print("💬 Generating backchannel...")
        reply = generate_backchannel(text, motion_type)
        print(f"Backchannel ({motion_type}): {reply}")

        print("🔊 Generating audio...")
        text_to_wav(reply, "reply.wav")

        print("▶️ Playing...")
        play_audio("reply.wav")

        print("--------------------------------")
    finally:
        is_processing_reaction = False




# =====================================
# ★ メイン：頭の動き監視スレッド起動
# =====================================
threading.Thread(target=head_motion_loop, daemon=True).start()

print("\n🎧 話しながら頷く / 首を振ると相槌が返ってきます！\n")

# メインスレッドは待機
while True:
    time.sleep(1)
