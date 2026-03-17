# Human Augmentation

この repo は、聞き手の頭部動作を IMU で取り、英語の相槌音声を返すデモと、評価用の実験コードをまとめたものです。

まず見る文書は次です。

- `docs/repository_guide.md`
- `docs/architecture/system_overview.md`

各ディレクトリには短い `README.md` を置いてあります。`app/`, `data/`, `scripts/`, `docs/` の README を辿ると、迷いにくいです。

## 開発の入口

依存は `uv` で入れます。

```bash
uv sync
```

学会デモの入口はこれです。台本なしの sensor-only モードで、英語相槌を返します。

```bash
bash scripts/demo-up.sh
```

直接 CLI を見るなら次です。

```bash
uv run python app/cli/demo.py --help
```

従来の transcript ベース実験の入口は次です。

```bash
uv run python app/cli/run.py --help
```

## 学会デモの最短手順

標準のデモ資産は `data/demo/` にあります。よく使うのは次です。

- `data/demo/catalog_en.tsv`
- `data/demo/backchannel_en/`
- `data/demo/scripts/conference_demo_en.json`
- `data/demo/script_audio/conference_demo_en/`

英語音声を OpenAI TTS で作り直すときは、`.env` に `OPENAI_API_KEY` を入れて次を実行します。

```bash
uv run python scripts/demo/generate_demo_audio.py --overwrite
```

`usbserial` 側を使うなら次です。

```bash
bash scripts/demo-up.sh
```

`xiao-bno055` 側を使うなら次です。

```bash
bash scripts/demo-up.sh demo-xiao-bno055
```

起動後は `Enter` で測定 ON/OFF、`q` で終了です。手動で鳴らすときは `1` から `7` が肯定系、`a` から `h` が否定系です。

ポートを固定したいときは、環境変数で渡せます。

```bash
DEMO_PORT=/dev/cu.usbserial-10 bash scripts/demo-up.sh
```

自動検出は `usbserial`、`wchusbserial`、`SLAB_USBtoUART`、`usbmodem` を見ます。外れたときは `DEMO_PORT` か `--port` を使ってください。

センサ単体モードでは `OPENAI_API_KEY` は要りません。詳細な配置は `docs/repository_guide.md` を見てください。
