#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

try:
    import matplotlib.pyplot as plt
    import pandas as pd
except ModuleNotFoundError as e:
    raise SystemExit("pandas と matplotlib が必要です。`uv sync --extra analysis` を先に実行してください。") from e


QUESTIONS: List[Tuple[str, str]] = [
    ("engaging", "Engaging"),
    ("understandable", "Understandable"),
    ("nervous", "Nervous"),
    ("exciting", "Exciting"),
    ("entertaining", "Entertaining"),
    ("competent", "Competent"),
    ("intent_conveyed", "Intent conveyed"),
]

CONDITION_ORDER = ["baseline", "system"]
CONDITION_LABEL = {
    "baseline": "No-Feedback\n(NF)",
    "system": "Auditory Feedback\n(AF)",
}


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    label: str
    path: Path
    scale_min: int
    scale_max: int


def _read_rows(path: Path) -> Tuple[List[str], List[dict]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows: List[dict] = []
        for row in reader:
            rows.append(dict(row))
    return fieldnames, rows


def _is_valid_marker(value: object) -> bool:
    v = str(value or "").strip()
    if not v:
        return False
    return v.lower() == "o" or v in {"○", "〇", "◯"}


def _infer_question_columns(fieldnames: List[str]) -> List[str]:
    excluded = {"タイムスタンプ", "氏名", "有効サンプル"}
    cols = [c for c in fieldnames if c not in excluded]
    if len(cols) != len(QUESTIONS):
        raise ValueError(
            f"質問列の数が想定と違います。想定={len(QUESTIONS)} 実際={len(cols)} columns={cols}"
        )
    return cols


def _to_dataframe(spec: DatasetSpec) -> pd.DataFrame:
    fieldnames, rows = _read_rows(spec.path)
    question_cols_raw = _infer_question_columns(fieldnames)
    raw_to_key = {raw: key for raw, (key, _label) in zip(question_cols_raw, QUESTIONS)}

    records = []
    for row in rows:
        rec = {
            "dataset": spec.key,
            "timestamp": row.get("タイムスタンプ", "").strip(),
            "participant": row.get("氏名", "").strip(),
            "is_valid": _is_valid_marker(row.get("有効サンプル", "")),
        }
        for raw_col, key in raw_to_key.items():
            v = (row.get(raw_col, "") or "").strip()
            rec[key] = float(v) if v != "" else None
        records.append(rec)

    df = pd.DataFrame.from_records(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y/%m/%d %H:%M:%S", errors="coerce")
    return df


def _extract_valid_pairs(df: pd.DataFrame) -> pd.DataFrame:
    valid = df[df["is_valid"]].copy()
    valid = valid.sort_values(["participant", "timestamp"], kind="stable").reset_index(drop=True)

    valid["within_participant_idx"] = valid.groupby("participant").cumcount()
    valid["pair_id"] = valid["within_participant_idx"] // 2
    valid["pos_in_pair"] = valid["within_participant_idx"] % 2
    valid["condition"] = valid["pos_in_pair"].map({0: "system", 1: "baseline"})

    paired = valid.groupby(["participant", "pair_id"]).filter(lambda g: len(g) == 2).copy()
    paired = paired.drop(columns=["within_participant_idx", "pos_in_pair"]).reset_index(drop=True)
    return paired


def _compute_total_scores(paired: pd.DataFrame, *, scale_min: int, scale_max: int) -> pd.DataFrame:
    paired = paired.copy()

    # 「緊張しているほど悪い」を「高いほど良い」に直す（例: 1..6 なら 1<->6, 2<->5 ...）
    paired["nervous_reversed"] = (scale_min + scale_max) - paired["nervous"]

    question_keys = [k for k, _ in QUESTIONS]
    total_cols = [c for c in question_keys if c != "nervous"] + ["nervous_reversed"]
    paired["total_score"] = paired[total_cols].sum(axis=1, skipna=False)
    return paired


def _to_total_long(paired: pd.DataFrame) -> pd.DataFrame:
    cols = ["dataset", "participant", "pair_id", "condition", "timestamp", "total_score"]
    return paired[cols].sort_values(["dataset", "participant", "pair_id", "condition"]).reset_index(drop=True)


def _to_total_diff(total_long: pd.DataFrame) -> pd.DataFrame:
    pivot = (
        total_long.pivot_table(
            index=["dataset", "participant", "pair_id"],
            columns="condition",
            values="total_score",
            aggfunc="first",
        )
        .reset_index()
        .copy()
    )
    pivot["diff_system_minus_baseline"] = pivot["system"] - pivot["baseline"]
    return pivot


def _to_long(paired: pd.DataFrame) -> pd.DataFrame:
    question_keys = [k for k, _ in QUESTIONS]
    long = paired.melt(
        id_vars=["dataset", "participant", "pair_id", "condition", "timestamp"],
        value_vars=question_keys,
        var_name="question",
        value_name="score",
    )
    return long.sort_values(["dataset", "question", "participant", "pair_id", "condition"]).reset_index(drop=True)


def _to_diff(long: pd.DataFrame) -> pd.DataFrame:
    pivot = (
        long.pivot_table(
            index=["dataset", "participant", "pair_id", "question"],
            columns="condition",
            values="score",
            aggfunc="first",
        )
        .reset_index()
        .copy()
    )
    pivot["diff_system_minus_baseline"] = pivot["system"] - pivot["baseline"]
    return pivot


def _question_label(key: str) -> str:
    for qk, label in QUESTIONS:
        if qk == key:
            return label
    return key


def _configure_plot_style(*, font_scale: float = 1.0) -> None:
    def s(v: float) -> float:
        return float(v) * float(font_scale)

    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": s(10),
            "axes.titlesize": s(10),
            "axes.labelsize": s(10),
            "xtick.labelsize": s(9),
            "ytick.labelsize": s(9),
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _plot_paired_small_multiples(
    long: pd.DataFrame,
    *,
    title: str,
    scale_min: int,
    scale_max: int,
    out_pdf: Path,
    out_png: Path,
) -> None:
    question_order = [k for k, _ in QUESTIONS]
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(12, 6), sharey=True)
    ax_list = list(axes.flat)

    for i, qkey in enumerate(question_order):
        ax = ax_list[i]
        sub = long[long["question"] == qkey]
        pivot = sub.pivot_table(index=["participant", "pair_id"], columns="condition", values="score", aggfunc="first")

        for (participant, pair_id), row in pivot.iterrows():
            if pd.isna(row.get("system")) or pd.isna(row.get("baseline")):
                continue
            ax.plot([0, 1], [row["baseline"], row["system"]], color="#888888", linewidth=1.0, alpha=0.7)
            ax.scatter([0, 1], [row["baseline"], row["system"]], s=28, color="#2a6fdb", zorder=3)

        mean_system = sub[sub["condition"] == "system"]["score"].mean()
        mean_baseline = sub[sub["condition"] == "baseline"]["score"].mean()
        if pd.notna(mean_system) and pd.notna(mean_baseline):
            ax.plot([0, 1], [mean_baseline, mean_system], color="#000000", linewidth=2.2, zorder=4)
            ax.scatter([0, 1], [mean_baseline, mean_system], s=40, color="#000000", zorder=5)

        ax.set_title(_question_label(qkey))
        ax.set_xticks([0, 1], [CONDITION_LABEL["baseline"], CONDITION_LABEL["system"]])
        ax.set_xlim(-0.06, 1.06)
        ax.set_ylim(scale_min - 0.5, scale_max + 0.5)
        ax.set_yticks(list(range(scale_min, scale_max + 1)))
        ax.grid(axis="y", color="#dddddd", linewidth=0.8)

    for j in range(len(question_order), len(ax_list)):
        ax_list[j].axis("off")

    fig.suptitle(title, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def _plot_differences(
    diff: pd.DataFrame,
    *,
    title: str,
    out_pdf: Path,
    out_png: Path,
) -> None:
    question_order = [k for k, _ in QUESTIONS]
    diff = diff.copy()
    diff["q_idx"] = diff["question"].map({k: i for i, k in enumerate(question_order)})
    diff = diff.sort_values(["q_idx", "participant", "pair_id"]).reset_index(drop=True)

    max_abs = diff["diff_system_minus_baseline"].abs().max()
    if pd.isna(max_abs):
        max_abs = 1.0
    max_abs = float(max(1.0, max_abs))

    fig, ax = plt.subplots(figsize=(12, 3.6))
    ax.axhline(0.0, color="#444444", linewidth=1.0, zorder=1)

    for i, qkey in enumerate(question_order):
        sub = diff[diff["question"] == qkey]
        x0 = float(i)
        xs = [x0 + (k - (len(sub) - 1) / 2) * 0.06 for k in range(len(sub))]
        ys = list(sub["diff_system_minus_baseline"])
        ax.scatter(xs, ys, s=35, color="#2a6fdb", zorder=3)

        mean = sub["diff_system_minus_baseline"].mean()
        if pd.isna(mean):
            mean = 0.0
        mean = float(mean)
        ax.scatter([x0], [mean], s=55, color="#000000", zorder=4)

    ax.set_xticks(list(range(len(question_order))), [_question_label(k) for k in question_order], rotation=20, ha="right")
    ax.set_ylabel("AF - NF")
    ax.set_title(title)
    ax.set_ylim(-max_abs - 0.5, max_abs + 0.5)
    ax.grid(axis="y", color="#dddddd", linewidth=0.8)

    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def _plot_total_paired(
    total_long: pd.DataFrame,
    *,
    title: str,
    out_pdf: Path,
    out_png: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    _draw_total_paired(ax, total_long, title=title, show_ylabel=True)

    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def _draw_total_paired(ax: plt.Axes, total_long: pd.DataFrame, *, title: str, show_ylabel: bool) -> None:
    pivot = total_long.pivot_table(index=["participant", "pair_id"], columns="condition", values="total_score", aggfunc="first")

    x_baseline = 0.0
    x_system = 1.35

    for (_participant, _pair_id), row in pivot.iterrows():
        if pd.isna(row.get("system")) or pd.isna(row.get("baseline")):
            continue
        ax.plot([x_baseline, x_system], [row["baseline"], row["system"]], color="#888888", linewidth=1.0, alpha=0.7)
        ax.scatter([x_baseline, x_system], [row["baseline"], row["system"]], s=45, color="#2a6fdb", zorder=3)

    mean_baseline = float(total_long[total_long["condition"] == "baseline"]["total_score"].mean())
    mean_system = float(total_long[total_long["condition"] == "system"]["total_score"].mean())
    ax.plot([x_baseline, x_system], [mean_baseline, mean_system], color="#000000", linewidth=2.2, zorder=4)
    ax.scatter([x_baseline, x_system], [mean_baseline, mean_system], s=65, color="#000000", zorder=5)

    ax.set_xticks([x_baseline, x_system], [CONDITION_LABEL["baseline"], CONDITION_LABEL["system"]])
    for lbl in ax.get_xticklabels():
        lbl.set_ha("center")
        lbl.set_multialignment("center")
        lbl.set_linespacing(1.15)
    ax.tick_params(axis="x", pad=8)
    if show_ylabel:
        ax.set_ylabel("Total score (higher is better)")
    ax.set_title(title, wrap=True)
    ax.grid(axis="y", color="#dddddd", linewidth=0.8)
    ax.set_xlim(x_baseline - 0.08, x_system + 0.08)


def _plot_total_paired_two(
    speaker_total_long: pd.DataFrame,
    listener_total_long: pd.DataFrame,
    *,
    out_pdf: Path,
    out_png: Path,
) -> None:
    all_vals = pd.concat([speaker_total_long["total_score"], listener_total_long["total_score"]], ignore_index=True)
    ymin = float(all_vals.min())
    ymax = float(all_vals.max())
    pad = max(1.0, (ymax - ymin) * 0.08)
    ymin -= pad
    ymax += pad

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(11.5, 4.2), sharey=True)
    _draw_total_paired(ax1, speaker_total_long, title="Speaker total (Nervous reversed)", show_ylabel=True)
    _draw_total_paired(ax2, listener_total_long, title="Listener total (Nervous reversed)", show_ylabel=False)

    ax1.set_ylim(ymin, ymax)
    ax2.set_ylim(ymin, ymax)

    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def _write_tsv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False)


def _summarize(long: pd.DataFrame, diff: pd.DataFrame) -> pd.DataFrame:
    by_cond = (
        long.groupby(["dataset", "condition", "question"], dropna=False)["score"]
        .agg(n="count", mean="mean", std="std", median="median")
        .reset_index()
    )
    by_diff = (
        diff.groupby(["dataset", "question"], dropna=False)["diff_system_minus_baseline"]
        .agg(n="count", mean="mean", std="std", median="median")
        .reset_index()
    )
    by_diff["condition"] = "system_minus_baseline"
    summary = pd.concat([by_cond, by_diff], ignore_index=True, sort=False)
    summary["question_label"] = summary["question"].map(_question_label)
    return summary.sort_values(["dataset", "question", "condition"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Speaker/Listener アンケートを可視化します（有効サンプルのみ）")
    parser.add_argument("--speaker", default="data/results/Speakerアンケート（回答） - AHs2026_Speaker.csv")
    parser.add_argument("--listener", default="data/results/Listenerアンケート（回答） - AHs2026_Listener.csv")
    parser.add_argument("--out-dir", default="data/figures/questionnaire")
    parser.add_argument("--out-tsv-dir", default="data/results")
    args = parser.parse_args()

    _configure_plot_style()

    specs = [
        DatasetSpec(
            key="speaker",
            label="Speaker (self-report)",
            path=Path(args.speaker),
            scale_min=1,
            scale_max=7,
        ),
        DatasetSpec(
            key="listener",
            label="Listener (rating)",
            path=Path(args.listener),
            scale_min=1,
            scale_max=7,
        ),
    ]

    out_dir = Path(args.out_dir)
    out_tsv_dir = Path(args.out_tsv_dir)

    all_long = []
    all_diff = []
    total_long_by_key: dict[str, pd.DataFrame] = {}

    for spec in specs:
        df = _to_dataframe(spec)
        paired = _extract_valid_pairs(df)
        n_valid_rows = int(df["is_valid"].sum())
        n_participants = int(paired["participant"].nunique())
        n_pairs = int(paired.groupby(["participant", "pair_id"]).ngroups)
        paired = _compute_total_scores(paired, scale_min=spec.scale_min, scale_max=spec.scale_max)

        total_long = _to_total_long(paired)
        total_diff = _to_total_diff(total_long)
        long = _to_long(paired)
        diff = _to_diff(long)

        _write_tsv(out_tsv_dir / f"questionnaire_{spec.key}_long.tsv", long)
        _write_tsv(out_tsv_dir / f"questionnaire_{spec.key}_diff.tsv", diff)
        _write_tsv(out_tsv_dir / f"questionnaire_{spec.key}_total_long.tsv", total_long)
        _write_tsv(out_tsv_dir / f"questionnaire_{spec.key}_total_diff.tsv", total_diff)

        print(f"{spec.key}: 有効行={n_valid_rows} ペア={n_pairs} 人={n_participants}")

        _plot_paired_small_multiples(
            long,
            title=f"{spec.label} (n={n_participants})",
            scale_min=spec.scale_min,
            scale_max=spec.scale_max,
            out_pdf=out_dir / f"{spec.key}_paired.pdf",
            out_png=out_dir / f"{spec.key}_paired.png",
        )
        _plot_differences(
            diff,
            title=f"{spec.label} differences (AF - NF)",
            out_pdf=out_dir / f"{spec.key}_diff.pdf",
            out_png=out_dir / f"{spec.key}_diff.png",
        )
        _plot_total_paired(
            total_long,
            title=f"{spec.label} total (Nervous reversed)",
            out_pdf=out_dir / f"{spec.key}_total_paired.pdf",
            out_png=out_dir / f"{spec.key}_total_paired.png",
        )

        all_long.append(long)
        all_diff.append(diff)
        total_long_by_key[spec.key] = total_long

    combined_long = pd.concat(all_long, ignore_index=True)
    combined_diff = pd.concat(all_diff, ignore_index=True)
    summary = _summarize(combined_long, combined_diff)
    _write_tsv(out_tsv_dir / "questionnaire_summary.tsv", summary)

    if "speaker" in total_long_by_key and "listener" in total_long_by_key:
        _plot_total_paired_two(
            total_long_by_key["speaker"],
            total_long_by_key["listener"],
            out_pdf=out_dir / "speaker_listener_total_paired.pdf",
            out_png=out_dir / "speaker_listener_total_paired.png",
        )

    print("OK")
    print(f"- figures: {out_dir}")
    print(f"- tsv: {out_tsv_dir / 'questionnaire_summary.tsv'}")


if __name__ == "__main__":
    main()
