#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

try:
    import pandas as pd
except ModuleNotFoundError as exc:
    raise SystemExit("pandas が必要です。`uv sync --extra analysis` で依存を追加してください。") from exc


def _load_aliases(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    df = pd.read_csv(path, sep="\t", header=None, names=["from", "to"], dtype="string", keep_default_na=False)
    mapping: dict[str, str] = {}
    for raw_from, raw_to in zip(df["from"], df["to"]):
        src = str(raw_from).strip()
        dst = str(raw_to).strip()
        if not src or not dst:
            continue
        mapping[src] = dst
    return mapping


def _prepare(
    *,
    input_path: Path,
    output_path: Path,
    ok_column: str,
    timestamp_col: str,
    name_col: str,
    alias_map: dict[str, str],
) -> None:
    df = pd.read_csv(input_path, encoding="utf-8-sig", keep_default_na=False)
    missing = [c for c in (timestamp_col, name_col, ok_column) if c not in df.columns]
    if missing:
        raise SystemExit(f"{input_path} に必要な列が見つかりません: {missing}")

    ok_vals = df[ok_column].astype("string").str.strip().str.lower()
    df = df[ok_vals == "ok"].copy()

    df[name_col] = df[name_col].astype("string").str.strip().map(lambda v: alias_map.get(str(v), str(v)))

    keep_cols = [timestamp_col, name_col]
    for col in list(df.columns):
        if col in keep_cols or col == ok_column:
            continue
        lowered = col.lower()
        if lowered.startswith("column") or col.startswith("列"):
            continue
        if lowered in {"reversed_nervousness", "score"}:
            continue
        keep_cols.append(col)

    df = df[keep_cols].copy()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Google Forms のエクスポートから v2 分析用CSVを作ります（Column 14 が ok の行だけ残します）"
    )
    parser.add_argument("--speaker-input", default="data/results/Speakerアンケート（回答） - AHs2026_Speaker (2).csv")
    parser.add_argument("--listener-input", default="data/results/Listenerアンケート（回答） - AHs2026_Listener (2).csv")
    parser.add_argument("--speaker-output", default="data/results/speaker_v2.csv")
    parser.add_argument("--listener-output", default="data/results/listener_v2.csv")
    parser.add_argument("--ok-column", default="Column 14")
    parser.add_argument("--timestamp-column", default="タイムスタンプ")
    parser.add_argument("--name-column", default="氏名")
    parser.add_argument("--name-aliases", default="data/results/name_aliases.tsv")
    args = parser.parse_args()

    alias_map = _load_aliases(Path(args.name_aliases))

    _prepare(
        input_path=Path(args.speaker_input),
        output_path=Path(args.speaker_output),
        ok_column=args.ok_column,
        timestamp_col=args.timestamp_column,
        name_col=args.name_column,
        alias_map=alias_map,
    )
    _prepare(
        input_path=Path(args.listener_input),
        output_path=Path(args.listener_output),
        ok_column=args.ok_column,
        timestamp_col=args.timestamp_column,
        name_col=args.name_column,
        alias_map=alias_map,
    )

    print("OK")
    print(f"- {args.speaker_output}")
    print(f"- {args.listener_output}")


if __name__ == "__main__":
    main()
