#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

try:
    import pandas as pd
except ModuleNotFoundError as exc:
    raise SystemExit("pandas が必要です。`uv sync --extra analysis` で依存を追加してください。") from exc


def _label_path(path: Path, *, name_col: str, timestamp_col: str, out_dir: Path) -> Path:
    df = pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False)
    if name_col not in df.columns:
        raise SystemExit(f"{path} に {name_col} 列が見つかりません")
    if timestamp_col not in df.columns:
        raise SystemExit(f"{path} に {timestamp_col} 列が見つかりません")

    df["__timestamp_value"] = pd.to_datetime(df[timestamp_col], format="%Y/%m/%d %H:%M:%S", errors="coerce")
    df = df.sort_values([name_col, "__timestamp_value"], kind="stable").reset_index(drop=True)
    df["session_index"] = df.groupby(name_col).cumcount()
    df["session_label"] = df["session_index"].map(lambda x: "system" if x % 2 == 0 else "baseline")
    df = df.drop(columns="__timestamp_value")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{path.stem}_labeled{path.suffix}"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="氏名ごとに先に現れた行を system、次を baseline とする列を追加します")
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=[
            "data/questionnaire/derived/anonymized/speaker_v2_anonymized.csv",
            "data/questionnaire/derived/anonymized/listener_v2_anonymized.csv",
        ],
    )
    parser.add_argument("--name-column", default="氏名")
    parser.add_argument("--timestamp-column", default="タイムスタンプ")
    parser.add_argument("--out-dir", default="data/questionnaire/derived/anonymized_labeled")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    outputs = []
    for input_path in args.inputs:
        path = Path(input_path)
        if not path.exists():
            raise SystemExit(f"{path} が見つかりません")
        outputs.append(_label_path(path, name_col=args.name_column, timestamp_col=args.timestamp_column, out_dir=out_dir))

    print("session_label を追記したファイルを作成しました")
    for o in outputs:
        print(f"- {o}")


if __name__ == "__main__":
    main()
