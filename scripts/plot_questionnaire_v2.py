#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import argparse
import sys

try:
    import matplotlib.pyplot as plt
    import pandas as pd
except ModuleNotFoundError as exc:
    raise SystemExit("pandas と matplotlib が必要です。`uv sync --extra analysis` で依存を追加してください。") from exc

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.plot_questionnaire import _configure_plot_style, _draw_total_paired, _to_total_diff


def _find_status_column(df: pd.DataFrame) -> str | None:
    candidates: list[str] = []
    for col in df.columns:
        lowered = col.lower()
        if lowered.startswith("column") or col.startswith("列") or "有効" in col or "valid" in lowered:
            candidates.append(col)

    best: str | None = None
    best_score = 0
    for col in candidates:
        vals = set(df[col].astype("string").str.strip().str.lower().unique().tolist())
        score = 2 if "ok" in vals else 0
        if "reject" in vals:
            score = max(score, 1)
        if score > best_score:
            best = col
            best_score = score
    return best


def _select_question_columns(columns: list[str], status_col: str, *, include_extra: bool) -> list[str]:
    excluded = {status_col, "session_index", "session_label"}

    def is_meta(col: str) -> bool:
        lowered = col.lower()
        if lowered.startswith("column") or lowered.startswith("列"):
            return True
        return lowered in {"reversed_nervousness", "score", "total_score", "nervous_reversed"}

    base = [c for c in columns[2:] if c not in excluded and not is_meta(c)]
    if include_extra:
        return base
    drop_keywords = ("意図", "intention", "reaction", "伝わって")
    return [col for col in base if not any(kw in col for kw in drop_keywords)]


def _load_total_long(
    path: Path,
    *,
    dataset_key: str,
    scale_min: int = 1,
    scale_max: int = 7,
    include_extra: bool = False,
) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False)
    if df.shape[1] < 4:
        raise SystemExit(f"{path} の列数が少なすぎます")

    status_col = _find_status_column(df)
    if status_col is not None:
        status_vals = df[status_col].astype("string").str.strip().str.lower()
        if (status_vals == "ok").any():
            df = df[status_vals == "ok"].copy()
        else:
            df = df[status_vals != "reject"].copy()
    else:
        status_col = ""

    original_columns = list(df.columns)
    if len(original_columns) < 6:
        raise SystemExit(f"{path} には列が足りません")
    question_cols = _select_question_columns(original_columns, status_col=status_col, include_extra=include_extra)
    if not question_cols:
        raise SystemExit(f"{path} で合計に使う質問列が見つかりませんでした")

    df["timestamp"] = pd.to_datetime(df["タイムスタンプ"], format="%Y/%m/%d %H:%M:%S", errors="coerce")
    df = df.dropna(subset=["timestamp", "氏名"]).reset_index(drop=True)

    # question_cols already computed from original columns

    df[question_cols] = df[question_cols].apply(pd.to_numeric, errors="coerce")
    df = df.copy()

    nervous_col = next((col for col in question_cols if "緊張" in col or "nervous" in col.lower()), question_cols[2])

    df["within_participant_idx"] = df.groupby("氏名").cumcount()
    df["pair_id"] = df["within_participant_idx"] // 2
    df["pos_in_pair"] = df["within_participant_idx"] % 2
    if "session_label" in df.columns:
        cond_vals = df["session_label"].astype("string").str.strip().str.lower()
        df["condition"] = cond_vals.map(lambda v: "system" if "system" in v else ("baseline" if "baseline" in v else None))
    else:
        df["condition"] = df["pos_in_pair"].map({0: "system", 1: "baseline"})

    df = df.dropna(subset=["condition"]).copy()

    paired = (
        df.groupby(["氏名", "pair_id"])
        .filter(lambda g: len(g) == 2 and set(g["condition"]) == {"system", "baseline"})
        .copy()
    )
    base_cols = [c for c in question_cols if c != nervous_col]

    paired["nervous_reversed"] = (scale_min + scale_max) - paired[nervous_col]
    paired["total_score"] = paired[base_cols].sum(axis=1, skipna=False) + paired["nervous_reversed"]
    paired = paired.dropna(subset=["total_score"]).copy()

    total_long = pd.DataFrame(
        {
            "dataset": dataset_key,
            "participant": paired["氏名"].astype(str),
            "pair_id": paired["pair_id"].astype(int),
            "condition": paired["condition"],
            "timestamp": paired["timestamp"],
            "total_score": paired["total_score"],
        }
    )
    return total_long.sort_values(["participant", "pair_id", "condition"]).reset_index(drop=True)


def _write_tsv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False)


def _plot_two_panels(speaker_long: pd.DataFrame, listener_long: pd.DataFrame, *, out_pdf: Path, out_png: Path) -> None:
    all_scores = pd.concat([speaker_long["total_score"], listener_long["total_score"]], ignore_index=True)
    if all_scores.empty:
        raise SystemExit("合計点データが空です")
    ymin = float(all_scores.min())
    ymax = float(all_scores.max())
    pad = max(1.0, (ymax - ymin) * 0.08)
    ymin -= pad
    ymax += pad

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(11.5, 4.2), sharey=True)
    _draw_total_paired(ax1, speaker_long, title="Speaker total (nervous reversed)", show_ylabel=True)
    _draw_total_paired(ax2, listener_long, title="Listener total (nervous reversed)", show_ylabel=False)

    ax1.set_ylim(ymin, ymax)
    ax2.set_ylim(ymin, ymax)

    # ラベル中心を軸中心に合わせる
    for ax in (ax1, ax2):
        for label in ax.get_xticklabels():
            label.set_ha("center")
            label.set_multialignment("center")

    # 中央で重ならないだけの間隔を確保する
    fig.tight_layout(pad=0.6, w_pad=0.9)
    fig.subplots_adjust(wspace=0.26)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def _resolve_path(preferred: str, fallback: str) -> Path:
    preferred_path = Path(preferred)
    if preferred_path.exists():
        return preferred_path
    fallback_path = Path(fallback)
    if fallback_path.exists():
        return fallback_path
    raise SystemExit(f"どちらのファイルも見つかりません: {preferred} / {fallback}")


def main() -> None:
    parser = argparse.ArgumentParser(description="v2 アンケートの合計点を speaker/listener 並列で表示します")
    parser.add_argument("--speaker", default="data/results/anonymized_labeled/speaker_v2_anonymized_labeled.csv")
    parser.add_argument("--listener", default="data/results/anonymized_labeled/listener_v2_anonymized_labeled.csv")
    parser.add_argument("--out-dir", default="data/figures/questionnaire")
    parser.add_argument("--out-stem", default="speaker_listener_total_paired_v3")
    parser.add_argument("--out-tsv-dir", default="data/results")
    args = parser.parse_args()

    _configure_plot_style(font_scale=1.40)

    speaker_path = _resolve_path(args.speaker, "data/results/anonymized/speaker_v2_anonymized.csv")
    listener_path = _resolve_path(args.listener, "data/results/anonymized/listener_v2_anonymized.csv")

    speaker_long = _load_total_long(speaker_path, dataset_key="speaker_v2")
    listener_long = _load_total_long(listener_path, dataset_key="listener_v2")
    speaker_long_extra = _load_total_long(speaker_path, dataset_key="speaker_v2", include_extra=True)
    listener_long_extra = _load_total_long(listener_path, dataset_key="listener_v2", include_extra=True)

    total_long = pd.concat([speaker_long, listener_long], ignore_index=True)
    total_diff = _to_total_diff(total_long)
    total_long_extra = pd.concat([speaker_long_extra, listener_long_extra], ignore_index=True)
    total_diff_extra = _to_total_diff(total_long_extra)

    _write_tsv(Path(args.out_tsv_dir) / "questionnaire_v2_total_long.tsv", total_long)
    _write_tsv(Path(args.out_tsv_dir) / "questionnaire_v2_total_diff.tsv", total_diff)
    _write_tsv(Path(args.out_tsv_dir) / "questionnaire_v2_total_long_with_intention.tsv", total_long_extra)
    _write_tsv(Path(args.out_tsv_dir) / "questionnaire_v2_total_diff_with_intention.tsv", total_diff_extra)

    out_dir = Path(args.out_dir)
    out_stem = str(args.out_stem)
    _plot_two_panels(
        speaker_long,
        listener_long,
        out_pdf=out_dir / f"{out_stem}.pdf",
        out_png=out_dir / f"{out_stem}.png",
    )
    _plot_two_panels(
        speaker_long_extra,
        listener_long_extra,
        out_pdf=out_dir / f"{out_stem}_with_intention.pdf",
        out_png=out_dir / f"{out_stem}_with_intention.png",
    )

    print("OK")
    print(f"- figure: {out_dir / f'{out_stem}.pdf'}")
    print(f"- figure extra: {out_dir / f'{out_stem}_with_intention.pdf'}")
    print(f"- tsv: {Path(args.out_tsv_dir) / 'questionnaire_v2_total_long.tsv'}")
    print(f"- tsv extra: {Path(args.out_tsv_dir) / 'questionnaire_v2_total_long_with_intention.tsv'}")


if __name__ == "__main__":
    main()
