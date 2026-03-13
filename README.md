# Human Augmentation

英語の相槌音声を返すデモ用の最小 README です。

## 英語音声の用意方法

標準の英語音声は、すでに次に入っています。

- `data/demo/catalog_en.tsv`
- `data/demo/backchannel_en/positive/`
- `data/demo/backchannel_en/negative/`

ふだんのセンサ単体モードでは、これだけあれば足ります。台本付きデモを使うときだけ、次も使います。

- `data/demo/scripts/conference_demo_en.json`
- `data/demo/script_audio/conference_demo_en/`

音声を作り直すときは、OpenAI TTS を使います。`.env` に `OPENAI_API_KEY` を入れて、次を実行してください。

```bash
uv run python scripts/generate_demo_audio.py --overwrite
```

## プログラムの起動方法

最初に依存を入れます。

```bash
uv sync
```

いちばん簡単な起動はこれです。台本は読み込まず、`usbserial` 側のメガネを自動で探して、センサだけで判断して英語音声を返します。

```bash
bash scripts/demo-up.sh
```

もう片方のメガネを使うときは、`device_id` を変えます。

```bash
bash scripts/demo-up.sh demo-xiao-bno055
```

起動したら、`Enter` で測定 ON/OFF、`q` で終了です。手動で鳴らすときは `1` から `7` が肯定系、`a` から `h` が否定系です。

ポートを固定したいときは、環境変数で渡せます。

```bash
DEMO_PORT=/dev/cu.usbserial-10 bash scripts/demo-up.sh
```

自動検出は `usbserial`、`wchusbserial`、`SLAB_USBtoUART`、`usbmodem` を見ます。外れたときは `DEMO_PORT` か `--port` を使ってください。

`demo-up.sh` を使わずに直接起動するなら次です。

```bash
uv run python app/cli/demo.py
```

台本付きデモに戻したいときだけ、`--script` を付けます。

```bash
uv run python app/cli/demo.py --script data/demo/scripts/conference_demo_en.json
```

センサ単体モードでは `OPENAI_API_KEY` は要りません。終了は `Ctrl+C` です。
