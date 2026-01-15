# 実装メモ（2026-01-14）

このドキュメントは、相槌システムの改善作業の記録と引き継ぎ用メモです。

---

## 研究の目的

**話し手の視点**：オンライン講義・会議で映像がないと「伝わっているか不安」になる。聞き手からのフィードバックがほしい。

**聞き手の視点**：寝ながら・掃除しながら聞いていても、「聞いてますよ」という合図を送りたい。

**LLMの役割**：聞き手の「意図」を汲み取り、話し手にとって好ましい情報だけを届ける。好ましくない状況（寝ている、よそ見している等）は伝えない。

**デバイス**：メガネ型、左耳の上のフレームにIMUを装着。基本は座って聞く想定。

---

## 今回実装した変更

### 1. 二段判断の構造（backchannel_graph.py）

```
prepare → decide → [should_respond=true] → choose → resolve
                 → [should_respond=false] → skip → END
```

- `decide`: LLMが「返すかどうか」を判断
- `choose`: 返す場合のみ、LLMが「何を返すか」を選択

### 2. 動きの特徴量を追加（signal.py）

| 特徴量                 | 説明                                  |
| ---------------------- | ------------------------------------- |
| `has_oscillation`      | 往復運動があるか（符号変化が1回以上） |
| `posture_returned`     | 姿勢が元に戻ったか                    |
| `is_symmetric`         | 正負の動きが対称か                    |
| `duration_s`           | 動きの持続時間                        |
| `nod_likelihood_score` | 頷きらしさスコア（0-6）               |
| `ratio_vs_5s/30s`      | 直近5秒/30秒との比較                  |

### 3. 区切れベースの判断（session.py）

- 以前：IMUサインが立った瞬間にLLMを呼ぶ
- 現在：文の区切れ（segment_end）のタイミングでLLMを呼ぶ

### 4. エピソード管理（signal_store.py）

- `SignalEpisode`: 1つの動きのまとまりを記録
- `consume_episodes()`: 区切れで消費
- 進行中のエピソードも含めて取得できるように修正

### 5. ジェスチャーキャリブレーション

`--gesture-calibration` オプションで、頷き・首振りの軸を自動推定。これにより、頷きを首振りと誤判定する問題が改善。

### 6. IMU閾値到達でトリガー（2026-01-15追加）

以前の問題：
- 1回のエピソードでもLLMを呼んでしまい、過剰に反応していた
- transcribe.txt の2分ごとの区切りがトリガーで、リアルタイム向きではなかった

変更後：
- 同じジェスチャー（nod/shake）が `min_gesture_count` 回（デフォルト3回）以上になった**瞬間に**LLMを呼ぶ
- transcribe.txt の区切りは無関係（コンテクスト提供のみ）
- 発火後は自動でリセットされ、再度3回溜まるまで発火しない

```
--min-gesture-count 3   # 3回以上で反応（デフォルト）
--min-gesture-count 2   # 2回以上で反応（敏感にしたい場合）
```

**追加した関数**（signal_store.py）:
- `count_by_gesture()`: ジェスチャーごとの発生回数をカウント
- `get_dominant_gesture()`: 最も多いジェスチャーを返す（閾値以上のみ）
- `set_threshold_callback()`: 閾値到達時のコールバックを設定
- `reset_threshold()`: 発火フラグをリセット

---

## 試したが戻したもの

### Chain of Thought（CoT）+ スコア化

- 分析ステップを入れて、各観点（動き、タイミング、文脈）をスコア化
- **問題**：LLM呼び出しが遅くなりすぎて非実用的
- **結果**：シンプルな `should_respond: true/false` に戻した

---

## 現在の課題

### 1. リアルタイム音声への対応

現在は事前に用意した `transcribe.txt` を使っているが、実際にはリアルタイムの音声を受け取る必要がある。

- 現在の仕組み：`[mm:ss]` でセグメントを区切り、その終わりで判断
- 問題：リアルタイムだと区切りが細かくなり、反応が過剰になる可能性

### 3. LLMの活用方法

現状、LLMに期待しているのは：
1. センサーデータの解釈（頷きか、単なる動きか）
2. 文脈との整合性判断
3. タイミングの判断
4. 相槌の選択



---

## 今後の方向性（案）

### 案A：階層的な判断

1. **即時判断（ルールベース）**：明らかに返さないケースを弾く
   - エピソードがない
   - クールダウン中
   - nod_score が低すぎる（例：2以下）

2. **遅延判断（LLM）**：返すべきか微妙なケースだけLLMに聞く
   - 一定間隔で「直近N秒のエピソードをまとめて評価」
   - 文脈を見て「今返すべきか」を判断

### 案B：イベント駆動 + バッファリング

1. エピソードを検出したらバッファに追加
2. 「返しても良いタイミング」（無音区間、文の終わりなど）を検出
3. タイミングが来たらバッファを評価して判断

### 案C：LLMの役割を絞る

- LLMは「相槌の種類選び」だけに特化
- 「返すかどうか」はルールベース（nod_score >= 4 など）

---

## ファイル構成

```
app/
├── agents/
│   └── backchannel_graph.py  # LLMエージェント（二段判断）
├── imu/
│   ├── signal.py             # 動きの特徴量抽出
│   ├── signal_store.py       # エピソード管理
│   ├── gesture_calibration.py # 頷き/首振りキャリブレーション
│   └── buffer.py             # IMUデータバッファ
├── runtime/
│   └── session.py            # メインセッション（区切れベース判断）
└── transcript/
    ├── speaker.py            # TTS読み上げ
    └── timeline.py           # 文字起こしタイムライン
```

---

## 起動コマンド

```bash
# 基本起動（キャリブレーション付き）
uv run python app/cli/run.py --ui --debug-agent --debug-signal --gesture-calibration

# キャリブレーションなし（素早く起動）
uv run python app/cli/run.py --ui --debug-agent --calibration-still-sec 0 --calibration-active-sec 0

# 敏感度を変更（2回でトリガー）
uv run python app/cli/run.py --ui --debug-agent --debug-signal --gesture-calibration --min-gesture-count 2
```

---

## 関連ドキュメント

- `docs/implementation_plan.md` - 当初の実装方針
- `docs/backchannel_timing.md` - タイミングに関する検討

