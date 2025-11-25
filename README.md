# human-augmentation (Head-Motion Backchannel AI)

## Overview
話しながら「頷き / 首振り」をすると、直前の相手の発話（最大8秒）をWhisperで文字起こしし、その内容に応じた自然な相槌を生成・読み上げするシステムです。
IMU（加速度・ジャイロ）で頭部動作を検出し、肯定 (nod)/否定 (shake)の相槌を分岐します。

## Features
- 常時録音 + 循環バッファ
- 直近 BUFFER_SECONDS 秒（デフォルト 8 秒）だけ保持
- 頭部動作でトリガ
- nod（頷き）→ 肯定・共感の相槌
- shake（首振り）→ 否定・違いを穏やかに受け止める相槌
- OpenAI Whisper + TTS
  - gpt-4o-transcribe で文字起こし
  - gpt-4o-mini で相槌生成
  - gpt-4o-mini-tts でWAV音声生成
- macOS afplay再生
- 再生中はバッファを一時停止し、再生後にバッファクリア

## Requirements
### OS
- macOS推奨（afplay を使用しているため）
- Windows/Linuxの場合は再生部分を変更すれば動作可能

### Hardware
- マイク入力可能なPC
- 頭部動作を計測できるIMUデバイス
- 例: ESP32 + MPU6050 / BNO055 等
- シリアル接続
- /dev/cu.usbserial-**** のようなデバイスが見えること

### Python
- Python 3.9 以上推奨 (3.13.5では動作確認済み)

## Installation
1. リポジトリ取得
```
git clone <this-repo>
cd <this-repo>
```

2. 仮想環境（任意）
```python3 -m venv .venv
source .venv/bin/activate
```

3. 依存ライブラリ
```
pip install -r requirements.txt
```

Note:
pyaudio が入らない場合、macOSでは以下が必要なことがあります。
```
brew install portaudio
pip install pyaudio
```

## OpenAI API Key 設定

環境変数 OPENAI_API_KEY を設定してください。

```
export OPENAI_API_KEY="sk-xxxx..."
```

永続化するなら .zshrc / .bashrc に追記。

## Configuration

プログラム冒頭の設定を必要に応じて変更してください。

```
PORT = "/dev/cu.usbserial-140"  # シリアルポート
BAUD = 115200                   # ボーレート
DT = 0.3                        # IMUデータ周期想定(秒)
THRESH_PITCH = 20               # shake判定しきい値
THRESH_YAW = 25                 # nod判定しきい値
COOLDOWN = 10.0                 # 連続検出のクールダウン(秒)

BUFFER_SECONDS = 8              # 直近バッファ秒数
RATE = 16000                    # 録音サンプリング周波数
CHUNK = 1024                    # 音声チャンクサイズ
MIN_SEC = 0.5                   # 最低録音秒数
```

## Usage

まず、ArduinoIDEでsample.inoのプログラムをマイコンに書き込みます。これにより、IMUセンサの計測値がシリアル通信に流れるようになります。

その後、以下のようにしてmain.pyを実行します。

```
python3 main.py
```

起動すると：
- マイク録音が開始されます
- IMUのシリアルを読み取り続けます
- 頭部動作が検出されると、直近の発話に相槌が返ってきます

例：
```
🎧 話しながら頷く / 首を振ると相槌が返ってきます！
🟢 NOD detected!
You said: 今日は研究がすごくうまくいって...
Backchannel (nod): それは良いですね、順調そうです！
▶️ Playing...
```

## IMU データフォーマット

シリアルで流れてくる1行は、少なくとも6つの数値を含む形式を想定しています。
```
ax, ay, az, gx, gy, gz
```

例（どんな区切りでもOK。正規表現で数値だけ抜き出します）：
```
0.01, -0.02, 1.00, 0.1, 0.5, -0.3
```

gy を pitch（shake検出用）

gz を yaw（nod検出用）

として積分・減衰して判定しています。

## Troubleshooting
1. PORT が違う / シリアルが開けない

接続中のポートを確認：

```
ls /dev/cu.*
```

表示されたものを PORT に設定してください。

2. 反応が多すぎる / 少なすぎる

THRESH_PITCH, THRESH_YAW を調整

DT が実際のIMU送信周期とズレていると誤検出します
→ IMU側の送信周期に合わせて変更してください。

3. 相槌が「空」になる

発話が短すぎる可能性
→ MIN_SEC を下げる / BUFFER_SECONDS を増やす

Whisperで無音判定されている可能性
→ マイク入力レベルを確認

4. afplay が無い

play_audio() を別の再生方法へ変更してください。

## Notes / Safety

本システムはリアルタイム音声を外部APIへ送信します。
録音や利用環境のプライバシーに注意して使用してください。

IMUの装着位置や座標系によって検出方向が反転する場合があります。
