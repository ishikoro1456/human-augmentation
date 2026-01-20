# Human Augmentation

ローカルで相槌を返す試作です。文字起こしを読み上げながら、IMUで「相槌の合図」（頷き/首振りなど）が出たときに、区切りを優先して相槌を選び、音声を再生します。

## 前提

- `data/catalog.tsv` に相槌の一覧があること
- `data/backchannel/` に音声ファイルがあること
- 環境変数 `OPENAI_API_KEY` を設定していること

## 起動

```
uv sync
uv run app/cli/run.py --port /dev/cu.usbserial-140 --baud 115200 --transcript transcribe.txt
```

## 2台Macでリアルタイム実験（マイク→文字起こし→相槌）

話し手（マイクあり）と、聞き手（IMUあり）を分けて動かすためのモードです。

- 話し手側: マイク音声を区切って文字起こしし、テキストを送ります
- 聞き手側: 受け取ったテキストと IMU を見て、相槌を選んで再生します

### 1) 聞き手側（IMU端末）を起動

同じネットワーク内から接続できるように、`--listen-host 0.0.0.0` を使います。

```
uv run python app/cli/listener.py --ui --trace-jsonl data/logs/trace_listener.jsonl --listen-host 0.0.0.0 --listen-port 8765 --port /dev/cu.usbserial-310 --baud 115200 --gesture-calibration --debug-agent --debug-signal
```

### 2) 話し手側（マイク端末）を起動

話し手側は `ffmpeg` が必要です。マイクの入力デバイスは、先に一覧を出して確認できます。

```
uv run python app/cli/talker.py --list-devices
```

聞き手側の IP を `--connect-host` に入れて起動します。`--mic-device` は環境に合わせてください。

```
uv run python app/cli/talker.py --connect-host 192.168.0.10 --connect-port 8765 --mic-device :0
```

### 2.5) 聞き手側の USB シリアル番号を探す

聞き手用 IMU が使うシリアルポートは `/dev/cu.*` に現れます。起動前に `ls /dev/cu.*` を実行し、`usbserial` や `usbmodem` などの名前を探してください。候補が複数ある場合は IMU を抜き差しする前後で `ls` を繰り返し、差分がデバイスです。必要なら `ls /dev/tty.*` も併せて確認し、`cu` 系と対応が取れるものを選びます。見つけたポート名を `--port` に渡して起動してください。

- 例: `ls /dev/cu.*` → `... /dev/cu.Bluetooth-Incoming-Port /dev/cu.usbserial-1420`
- IMU が `usbserial` 以外の名前になることもあるので、`tty` 側と合わせて候補を確認すると安全です
- 起動例: `uv run python app/cli/listener.py --port /dev/cu.usbserial-1420 --baud 115200 --listen-host 0.0.0.0 --listen-port 8765 ...`

### 3) ログの見方

- 聞き手側の `--trace-jsonl` に、IMU、渡した文脈、モデルの判断（理由を含む）、再生結果が残ります
- 話し手側は `data/stt_segments/` に区切った音声（wav）を残します（文字起こしの失敗確認に使えます）

起動直後に IMU の計測フェーズがあります。急いで試したい場合は次でスキップできます。

```
uv run app/cli/run.py --calibration-still-sec 0 --calibration-active-sec 0
```

計測が忙しく感じる場合は、開始前/フェーズ間/計測後の待ち時間を増やせます。

```
uv run app/cli/run.py --calibration-start-delay-sec 5 --calibration-between-sec 5 --startup-wait-sec 3
```

頷き/首振りの「弱い・強い」の差分も最初に覚えさせたい場合は、次を付けます。

```
uv run app/cli/run.py --gesture-calibration
```

`transcribe.txt` は OpenAI の TTS で順に読み上げます。読み上げチャンクが終わるたびに、その時点までの文字起こしコンテクストと IMU を LLM に渡します。([platform.openai.com](https://platform.openai.com/docs/guides/text-to-speech?utm_source=openai))

いまの実装では、相槌の判断点は基本的に「読み上げチャンクの区切り（segment_end）」です。ただし、区切りが来ない状態が長く続くことがあるので、合図が出たら最大 `--human-signal-hold-sec` 秒のあいだ保留し、締め切り直前にも一度だけ判断します（区切りではないこともモデルに伝えます）。

相槌の音声は別チャンネルで再生します。読み上げ音声と重なることはありますが、割り込みを避けるために区切りを優先します。

相槌は、IMU から「相槌を出したいサイン」が見えたときだけ出します。何もしていないのに勝手に相槌が増えるのを避けるためです。サインの判定自体は常時回しておき、相槌の判断タイミングで直近のサインを参照します（保持時間は調整できます）。

```
uv run app/cli/run.py --no-require-human-signal
```

サインが出てから反応するまでの猶予（クールダウンや遅延の吸収）を調整したい場合は次を付けます。

```
uv run app/cli/run.py --human-signal-hold-sec 1.2
```

TTS の音声は `data/tts_cache/` に保存します。`transcribe.txt` の内容や TTS の設定が変わると、別ファイルとして生成します。

起動中の状況を見やすくするには、ダッシュボード表示を付けます。

```
uv run app/cli/run.py --ui
```

文字起こしの内容を確認したい場合は次を付けます。

```
uv run app/cli/run.py --debug-transcript
```

IMU と選択理由を確認したい場合は次を付けます。

```
uv run app/cli/run.py --debug-agent
```

IMUの「相槌サイン」判定（サインあり/なしと閾値）を確認したい場合は次を付けます。

```
uv run app/cli/run.py --debug-signal
```

首振り(否定)と頷き(肯定)の向きが混ざる場合は、IMUの取り付け向きと軸の対応が合っていない可能性があります。`--debug-signal` の `axis=` を見ながら、次の軸設定を調整できます。

```
uv run app/cli/run.py --imu-nod-axis gy --imu-shake-axis gz
```
