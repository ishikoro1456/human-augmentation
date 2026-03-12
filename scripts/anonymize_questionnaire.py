#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Mapping

try:
    import pandas as pd
except ModuleNotFoundError as exc:
    raise SystemExit("pandas が必要です。`uv sync --extra analysis` で依存を追加してください。") from exc


def _load_names(path: Path, name_col: str) -> Iterable[str]:
    df = pd.read_csv(path, encoding="utf-8-sig", dtype={name_col: "string"}, keep_default_na=False)
    for name in df[name_col]:
        if not name or isinstance(name, float):
            continue
        yield str(name).strip()


def _build_mapping(paths: list[Path], name_col: str, prefix: str, start: int) -> Mapping[str, str]:
    mapping: dict[str, str] = {}
    counter = start
    for path in paths:
        for raw in _load_names(path, name_col):
            if not raw or raw in mapping:
                continue
            mapping[raw] = f"{prefix}{counter:02d}"
            counter += 1
    return mapping


def _anonymize(path: Path, name_col: str, mapping: Mapping[str, str], out_dir: Path) -> Path:
    df = pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False)
    if name_col not in df.columns:
        raise ValueError(f"{path} に {name_col} 列が見つかりません")
    df[name_col] = (
        df[name_col]
        .astype("string")
        .apply(lambda v: mapping.get(str(v).strip(), "PXX") if v is not None else "PXX")
        .fillna("PXX")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{path.stem}_anonymized{path.suffix}"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="プレ／聞き手アンケートの氏名を番号で置き換えます")
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=[
            "data/results/speaker_v2.csv",
            "data/results/listener_v2.csv",
        ],
    )
    parser.add_argument("--name-column", default="氏名")
    parser.add_argument("--out-dir", default="data/results/anonymized")
    parser.add_argument("--prefix", default="P")
    parser.add_argument("--start-index", type=int, default=1)
    args = parser.parse_args()

    paths = [Path(p) for p in args.inputs]
    mapping = _build_mapping(paths, args.name_column, args.prefix, args.start_index)
    if not mapping:
        raise SystemExit("氏名が見つからず、マッピングできませんでした。")

    outputs = []
    for path in paths:
        out_path = _anonymize(path, args.name_column, mapping, Path(args.out_dir))
        outputs.append(out_path)

    print(f"{len(mapping)}人分を {args.prefix}NN 形式で置き換えました")
    for out in outputs:
        print(f"- {out}")


if __name__ == "__main__":
    main()
