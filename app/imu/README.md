# app/imu

IMU の読み取りと正規化です。上位の説明は [../README.md](../README.md)、全体像は [../../docs/repository_guide.md](../../docs/repository_guide.md) を見ます。

- ここにあるもの: シリアル読み取り、デバイス差分吸収、ジェスチャ検出です。
- 最初に見るファイル: `device.py`, `parser.py`, `signal.py`
- 更新してよいもの: IMU 入力と検出ロジックです。
- 生成物: ありません。
- 検索語: `six_axis`, `gyro_xyz`, `gesture`, `calibration`

入力フォーマットの基本は 1 行に `ax, ay, az, gx, gy, gz` です。区切り文字は固定しません。
