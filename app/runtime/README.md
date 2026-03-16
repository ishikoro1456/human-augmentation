# app/runtime

従来の listener と talker の実行制御です。入口は [../../README.md](../../README.md)、配置は [../../docs/repository_guide.md](../../docs/repository_guide.md) を見ます。

- ここにあるもの: セッション管理、状態表示、trace 保存です。
- 最初に見るファイル: `session.py`, `listener_session.py`, `trace.py`
- 更新してよいもの: 実験系の実行フローです。
- 生成物: 実行ログは `data/runtime/` に出ます。
- 検索語: `listener`, `session`, `trace`, `status`
