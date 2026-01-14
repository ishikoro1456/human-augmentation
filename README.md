# Human Augmentation

ローカルで相槌を返す試作です。文字起こしを読み上げながら、読み上げチャンクの直後に、その時点までの文字起こしと IMU の情報を使って相槌を選び、音声を再生します。

## 前提

- `data/catalog.tsv` に相槌の一覧があること
- `data/backchannel/` に音声ファイルがあること
- 環境変数 `OPENAI_API_KEY` を設定していること

## 起動

```
uv sync
uv run app/cli/run.py --port /dev/cu.usbserial-140 --baud 115200 --transcript transcribe.txt
```

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

いまの実装では、相槌の判断点を「読み上げチャンクの直後」に絞っています。各チャンクが終わるたびに、その時点までの文字起こしコンテクストと IMU を LLM に渡して、相槌を選びます。

相槌の音声は別チャンネルで再生します。相槌の再生中は次の読み上げを待たせるので、話し手の音声と重なりにくいです。

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
uv run app/cli/run.py --imu-nod-axis gy --imu-shake-axis gz --imu-tilt-axis gx
```
