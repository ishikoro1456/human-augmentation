# Human Augmentation

2台の端末を使って、リアルタイムで相槌を返す実験をするための試作です。

基本は、聞き手側を macOS（IMUをつなぐ端末）、話し手側を macOS または Windows（マイクの端末）で動かします。

今回の実験では `app/cli/listener.py` と `app/cli/talker.py` を使います。`app/cli/run.py` は使いません。

## 前提

この実験に必要なものを、短くまとめます。

- 2台の端末（話し手: マイク、聞き手: IMU）
- 両方の端末にこのリポジトリがあり、`uv sync` が通る
- 聞き手側（IMU端末）で `OPENAI_API_KEY` を設定している（`.env` でも可）
- 話し手側で `ffmpeg` が実行できる
- 聞き手側で話し手の音声を聞きたい場合は `ffplay` が実行できる（`ffmpeg` を入れると一緒に入ります）
- `data/catalog.tsv` と `data/backchannel/` がある（相槌の候補と音声）
- 今回（頷き/首振り）のカタログは `positive` / `negative` の2種類だけを使います（音声は `data/backchannel/positive/` と `data/backchannel/negative/`）

## 何をどちらで起動するか

聞き手側（IMU端末。macOS想定）では、IMU を読みつつ、話し手の音声を受け取って再生します。同時に、受け取った音声から文字起こしを作って文脈をため、相槌を決めて話し手側へ送ります（listener）。

話し手側（マイク端末。macOS または Windows）では、マイク音声をそのまま聞き手へ送り、聞き手から届いた相槌を再生します（talker）。

## 手順（今回の実験）

### 0) 両方の端末で準備

両方の端末で、最初に依存を入れます。

```
uv sync
```

### 1) 聞き手側（IMU端末）を起動する

まず IMU の USB シリアル番号を確認します。次を実行して、`usbserial` や `usbmodem` などの名前を探してください。

```
ls /dev/cu.*
```

候補が複数ある場合は、IMU を抜く前と刺した後で `ls /dev/cu.*` を繰り返し、差分が IMU です。必要なら `ls /dev/tty.*` も見てください。

次に listener を起動します。`--listen-host 0.0.0.0` にすると、同じネットワーク内の別の端末から接続できます。

```
uv run python app/cli/listener.py --ui --trace-jsonl data/logs/trace_listener.jsonl --listen-host 0.0.0.0 --listen-port 8765 --port /dev/cu.usbserial-310 --baud 115200
```

`--ui` は見やすさ重視の画面（participant）を出します。開発中に細かい情報も見たいときは、`--ui-mode debug` を付けます。

はじめて動かすときに IMU の向きが分からない場合は、追加で `--gesture-calibration` を付けると、軸の推定が入ります。

起動直後に IMU の計測が入ります。急いで試すだけなら、次で静止/動作の計測を省略できます。

```
uv run python app/cli/listener.py --ui --trace-jsonl data/logs/trace_listener.jsonl --listen-host 0.0.0.0 --listen-port 8765 --port /dev/cu.usbserial-310 --baud 115200 --calibration-still-sec 0 --calibration-active-sec 0
```

相槌が出ない、または出すぎるなどのときは、上のコマンドの末尾に `--debug-agent --debug-signal` を足すと理由が見やすくなります。

相槌の比較モードを変える場合は `--mode` を使います。

- `--mode llm`（既定）: IMU とモデルで相槌を決めます
- `--mode human`: 端末に表示される一覧を見て、番号（または id）を入力して Enter で確定します
- `--mode none`: 相槌を返しません

### 2) 話し手側（マイク端末）を起動する

聞き手側の IP を調べます。聞き手側（macOS）の端末で次を実行し、出てきた IP を控えてください（Wi-Fi の場合）。

```
ipconfig getifaddr en0
```

空なら `en1` を試してください。

ここで出た IP は「聞き手側の IP」です。話し手側の IP と同じになるとは限りません。talker の `--connect-host` には、この聞き手側の IP を入れてください。

次に、話し手側でマイクの番号を確認します。

macOS の場合は次です。

```
uv run python app/cli/talker.py --list-devices
```

この表示のうち、`AVFoundation audio devices` の番号が音声の入力です。内蔵マイクを使いたい場合は、そこに出ている `MacBook Proのマイク` などの番号を使います。

Windows の場合は次です。

```
uv run python app/cli/talker.py --mic-backend dshow --list-devices
```

WSL 上で動かす場合は、Windows ではなく Linux として動きます。まずは次を試してください。

```
uv run python app/cli/talker.py --mic-backend pulse --list-devices
```

`pactl` が無いと言われた場合は、WSL 側に入っているか確認してください。うまくいかないときは、talker は WSL ではなく Windows 側（PowerShell など）で動かすのが早いことがあります。

起動するときの `--mic-device` は `:<音声番号>` です。たとえば音声番号が 2 なら `:2` です。

macOS の例です。

```
uv run python app/cli/talker.py --connect-host 192.168.0.10 --connect-port 8765 --mic-device :2 --trace-jsonl data/logs/trace_talker.jsonl
```

Windows の例です（`--mic-backend dshow` を付けます）。

```
uv run python app/cli/talker.py --connect-host 192.168.0.10 --connect-port 8765 --mic-backend dshow --mic-device :0 --trace-jsonl data/logs/trace_talker.jsonl
```

WSL の例です。

```
uv run python app/cli/talker.py --connect-host 192.168.0.10 --connect-port 8765 --mic-backend pulse --mic-device default --trace-jsonl data/logs/trace_talker.jsonl
```

talker は、接続できるまで待ってからマイク取り込みを始めます。先に listener を起動してください。

### 3) うまく動いているかの目安

聞き手側は、話し手の声が聞こえ、ダッシュボード（`--ui`）に直近の文字起こしと、判断が出ます。話し手側は、聞き手が選んだ相槌が再生されます。必要なら talker に `--debug-net` を付けると、受け取った相槌の id が表示されます。

### 4) 比較の回し方（none / human / llm）

このリポジトリでは、listener の `--mode` を変えて比較します。どのモードでも、talker は同じコマンドで動かせます。

- `--mode none` は、相槌を返さないので「聞き手がいない」に近い状態になります
- `--mode human` は、一覧からキー入力で相槌を返します（生声ではなくカタログです）
- `--mode llm` は、IMU とモデルで相槌を決めます（既定）

### 5) ログを残す

listener の `--trace-jsonl` には、IMU、渡した文脈、モデルの理由、送った相槌などが残ります。talker の `--trace-jsonl` には、受け取った相槌と再生結果が残ります。

2つのログは、同じ `experiment_id` が入るので、あとで突合できます。これは listener が自動で作って talker に知らせます（必要なら listener に `--experiment-id` を付けて固定できます）。

聞き手側は `data/stt_segments_listener/` に、文字起こしに使った区切り音声（wav）が残ります。

相槌の一覧だけを見たいときは、次で TSV に要約できます（ログを同じ場所に置いてから実行してください）。

```
uv run python scripts/trace_to_tsv.py data/logs/trace_listener.jsonl data/logs/trace_talker.jsonl --out data/logs/backchannel_summary.tsv
```

止めるときは、どちらも `Ctrl+C` です。
