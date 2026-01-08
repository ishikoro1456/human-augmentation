# Human Augmentation

ローカルで相槌を返す試作です。IMU の反応が来たときに、直近の発話と IMU の情報を使って相槌を選び、音声を再生します。

## 前提

- `data/catalog.tsv` に相槌の一覧があること
- `data/backchannel/` に音声ファイルがあること
- 環境変数 `OPENAI_API_KEY` を設定していること

## 起動

```
uv sync
uv run app/cli/run.py --port /dev/cu.usbserial-140 --baud 115200 --transcript transcribe.txt
```

`transcribe.txt` は OpenAI の TTS で順に読み上げます。IMU の反応が来ると、その時点までに読み上げた全文を LLM に渡します。([platform.openai.com](https://platform.openai.com/docs/guides/text-to-speech?utm_source=openai))

TTS の音声は `data/tts_cache/` に保存します。次回以降は再生成しません。

文字起こしの内容を確認したい場合は次を付けます。

```
uv run app/cli/run.py --debug-transcript
```

IMU と選択理由を確認したい場合は次を付けます。

```
uv run app/cli/run.py --debug-agent
```
