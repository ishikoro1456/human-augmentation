# Repository Guide

この文書は、どこに何を置くかを迷わないための見取り図です。よく触る場所だけを先に書き、細かい補足は後ろへ回しています。

## まず見る場所

- `README.md`
  学会デモをすぐ起動するときの入口です。
- 各ディレクトリの `README.md`
  その場所で何を触ってよいかを短く書いています。
- `app/`
  実装本体です。CLI、デモ実行、IMU 処理、評価画面が入っています。
- `data/demo/`
  学会デモで使う英語音声、台本、デバイス設定を置きます。
- `docs/`
  設計メモと repo の案内を置きます。

## app の役割

- `app/cli/`
  実行入口です。学会デモは `app/cli/demo.py`、従来の実験系は `app/cli/run.py` を使います。
- `app/demo/`
  学会デモ用のセッション制御です。静的台本、相槌再生、手動キー再生、測定 ON/OFF をここで扱います。
- `app/imu/`
  IMU の読み取り、正規化、ジェスチャ検出を置きます。
- `app/eval/`
  実験後評価の画面、保存、集計の土台です。
- `app/runtime/`
  従来の listener と talker のセッション制御です。
- `app/audio/`, `app/transcript/`, `app/tts/`
  音声再生、文字起こし再生、TTS をまとめています。

## data の置き場所

- `data/demo/`
  学会デモ専用です。`scripts/`, `script_audio/`, `backchannel_en/`, `devices.json` を含みます。
- `data/transcripts/`
  読み上げ用 transcript の元データです。root に置かず、ここに集約します。
- `data/scripts/`
  既存の実験用 script JSON を置きます。
- `data/questionnaire/raw/`
  アンケートの元データです。手入力の置換表もここに置きます。
- `data/questionnaire/derived/`
  集計や匿名化の派生物です。再生成を前提にし、git には基本入れません。
- `data/runtime/`
  logs、STT segments、TTS cache、評価 DB などの実行時生成物です。git には入れません。

## docs と paper_materials

- `docs/`
  長めの説明を置きます。`architecture/` は全体像、`notes/` は検討メモです。
- `paper_materials/`
  論文や発表向けの素材です。論文 PDF、動画 text、最終図表をここに置きます。
- `archive/reference/`
  過去の試作や参考実装です。日常開発の導線から外しています。

## scripts の整理方針

- `scripts/demo-up.sh`
  学会デモの公開入口です。中身は `scripts/demo/demo-up.sh` にあります。
- `scripts/demo/`
  デモ用英語音声を OpenAI TTS で再生成します。
- `scripts/transcripts/`
  transcript の整形です。入出力の既定値は `data/transcripts/` にそろえます。
- `scripts/eval/`
  評価集計、匿名化、図表生成です。
- `scripts/paper/`
  発表動画など論文周辺の生成スクリプトです。

## 検索の前提

- 普段の `rg` は `.rgignore` を通して使います。
  `archive/`, `data/runtime/`, `data/questionnaire/derived/`, `paper_materials/video/generated/` は既定検索から外しています。
- まず `app/`, `scripts/`, `docs/`, `tests/`, `firmware/`, `data/demo/`, `data/transcripts/` を見ます。
- 参考置き場まで探したいときは、`rg -uu` で明示して検索します。

## root に置くもの

root には、できるだけ次だけを残します。

- `README.md`
- `pyproject.toml`
- `uv.lock`
- `AGENTS.md`

それ以外の文書、transcript、論文 PDF は、それぞれ `docs/`, `data/`, `paper_materials/` に置きます。
