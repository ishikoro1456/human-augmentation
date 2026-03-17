#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import math
from dataclasses import dataclass
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
except ModuleNotFoundError as exc:
    raise SystemExit("pandas と matplotlib が必要です。`uv sync --extra analysis` で依存を追加してください。") from exc


@dataclass(frozen=True)
class SummaryRow:
    variant: str
    dataset: str
    n_pairs: int
    n_nonzero: int
    mean_baseline: float
    mean_system: float
    mean_diff: float
    mean_ci_low: float
    mean_ci_high: float
    median_diff: float
    median_ci_low: float
    median_ci_high: float
    std_diff: float
    dz: float
    n_pos: int
    n_neg: int
    n_zero: int
    wilcoxon_w_plus: float
    wilcoxon_p_exact: float
    p_signflip_mean: float
    p_sign_test: float


def _load_diff_tsv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    required = {"dataset", "participant", "pair_id", "baseline", "system", "diff_system_minus_baseline"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(f"{path} に必要な列が足りません: {missing}")
    return df


def _bootstrap_ci(values: np.ndarray, *, fn, n_boot: int, seed: int) -> tuple[float, float]:
    if len(values) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        samp = rng.choice(values, size=len(values), replace=True)
        boots[i] = float(fn(samp))
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return (float(lo), float(hi))


def _sign_test_pvalue(diffs: np.ndarray) -> float:
    diffs = [float(x) for x in diffs if x != 0.0 and not math.isnan(float(x))]
    n = len(diffs)
    if n == 0:
        return 1.0
    k = sum(1 for x in diffs if x > 0.0)
    m = min(k, n - k)
    p_one_side = sum(math.comb(n, i) * (0.5**n) for i in range(m + 1))
    return float(min(1.0, 2.0 * p_one_side))


def _wilcoxon_signed_rank_exact(diffs: np.ndarray) -> tuple[float, float, int]:
    diffs = np.asarray(diffs, dtype=float)
    diffs = diffs[~np.isnan(diffs)]
    diffs = diffs[diffs != 0.0]
    n = int(len(diffs))
    if n == 0:
        return (0.0, 1.0, 0)

    abs_diffs = np.abs(diffs)
    ranks = pd.Series(abs_diffs).rank(method="average").to_numpy(dtype=float)
    ranks_scaled = np.rint(ranks * 2.0).astype(int)

    obs = int(ranks_scaled[diffs > 0.0].sum())
    total_sum = int(ranks_scaled.sum())

    dp = np.zeros(total_sum + 1, dtype=np.int64)
    dp[0] = 1
    for r in ranks_scaled.tolist():
        for s in range(total_sum, r - 1, -1):
            dp[s] += dp[s - r]

    total = float(2**n)
    p_low = float(dp[: obs + 1].sum() / total)
    p_high = float(dp[obs:].sum() / total)
    p = min(1.0, 2.0 * min(p_low, p_high))

    return (obs / 2.0, p, n)


def _signflip_mean_pvalue_exact(diffs: np.ndarray) -> float:
    diffs = np.asarray(diffs, dtype=float)
    if len(diffs) == 0:
        return 1.0
    obs = float(np.mean(diffs))
    if obs == 0.0:
        return 1.0

    n = len(diffs)
    total = 2**n
    count = 0
    threshold = abs(obs) - 1e-12

    for signs in itertools.product((1.0, -1.0), repeat=n):
        perm_mean = float(np.mean(diffs * np.asarray(signs)))
        if abs(perm_mean) >= threshold:
            count += 1
    return float(count / total)


def _summarize_variant(diff_df: pd.DataFrame, *, variant: str, n_boot: int, seed: int) -> list[SummaryRow]:
    rows: list[SummaryRow] = []
    for dataset in sorted(diff_df["dataset"].unique()):
        sub = diff_df[diff_df["dataset"] == dataset].copy()
        baseline = sub["baseline"].astype(float).to_numpy()
        system = sub["system"].astype(float).to_numpy()
        diffs = sub["diff_system_minus_baseline"].astype(float).to_numpy()

        n_pairs = int(len(diffs))
        if n_pairs == 0:
            continue

        mean_baseline = float(np.mean(baseline))
        mean_system = float(np.mean(system))
        mean_diff = float(np.mean(diffs))
        median_diff = float(np.median(diffs))
        std_diff = float(np.std(diffs, ddof=1)) if n_pairs > 1 else float("nan")
        dz = float(mean_diff / std_diff) if n_pairs > 1 and std_diff > 0 else float("nan")

        mean_ci_low, mean_ci_high = _bootstrap_ci(diffs, fn=np.mean, n_boot=n_boot, seed=seed)
        median_ci_low, median_ci_high = _bootstrap_ci(diffs, fn=np.median, n_boot=n_boot, seed=seed + 1)

        n_pos = int(np.sum(diffs > 0))
        n_neg = int(np.sum(diffs < 0))
        n_zero = int(np.sum(diffs == 0))

        wilcoxon_w_plus, wilcoxon_p_exact, n_nonzero = _wilcoxon_signed_rank_exact(diffs)

        p_signflip_mean = _signflip_mean_pvalue_exact(diffs)
        p_sign_test = _sign_test_pvalue(diffs)

        rows.append(
            SummaryRow(
                variant=variant,
                dataset=str(dataset),
                n_pairs=n_pairs,
                n_nonzero=n_nonzero,
                mean_baseline=mean_baseline,
                mean_system=mean_system,
                mean_diff=mean_diff,
                mean_ci_low=mean_ci_low,
                mean_ci_high=mean_ci_high,
                median_diff=median_diff,
                median_ci_low=median_ci_low,
                median_ci_high=median_ci_high,
                std_diff=std_diff,
                dz=dz,
                n_pos=n_pos,
                n_neg=n_neg,
                n_zero=n_zero,
                wilcoxon_w_plus=float(wilcoxon_w_plus),
                wilcoxon_p_exact=float(wilcoxon_p_exact),
                p_signflip_mean=p_signflip_mean,
                p_sign_test=p_sign_test,
            )
        )
    return rows


def _write_tsv(path: Path, rows: list[SummaryRow]) -> None:
    df = pd.DataFrame([r.__dict__ for r in rows])
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False)


def _format_row_for_md(r: SummaryRow) -> str:
    std_diff = "NA" if math.isnan(r.std_diff) else f"{r.std_diff:.2f}"
    dz = "NA" if math.isnan(r.dz) else f"{r.dz:.2f}"
    return (
        f"- n={r.n_pairs} (non-zero diffs={r.n_nonzero}) / baseline mean={r.mean_baseline:.2f} / system mean={r.mean_system:.2f}\n"
        f"  - diff (AF-NF): mean={r.mean_diff:.2f} (95% CI {r.mean_ci_low:.2f}..{r.mean_ci_high:.2f}), "
        f"median={r.median_diff:.2f} (95% CI {r.median_ci_low:.2f}..{r.median_ci_high:.2f})\n"
        f"  - +:{r.n_pos} 0:{r.n_zero} -:{r.n_neg} / std(diff)={std_diff} / dz={dz}\n"
        f"  - Wilcoxon signed-rank (exact): W+={r.wilcoxon_w_plus:.1f} / p={r.wilcoxon_p_exact:.3f}\n"
        f"  - p(sign-flip, mean)={r.p_signflip_mean:.3f} / p(sign test)={r.p_sign_test:.3f}"
    )


def _write_markdown(path: Path, rows: list[SummaryRow], *, without_path: Path, with_path: Path) -> None:
    by_variant: dict[str, list[SummaryRow]] = {}
    for r in rows:
        by_variant.setdefault(r.variant, []).append(r)

    lines: list[str] = []
    lines.append("# Questionnaire v2: total score statistics\n")
    lines.append("このファイルは、Auditory Feedback (AF) と No-Feedback (NF) の差を数字で眺めるためのメモです。\n")
    lines.append("計算の前提は次のとおりです。\n")
    lines.append("- 有効行だけを使う（元データでは Column 14 が ok の行だけ）\n- 緊張は逆スコア（1↔7）\n- 差は AF-NF\n")
    lines.append(f"入力TSV: {without_path} / {with_path}\n")

    for variant in ("without_intention", "with_intention"):
        if variant not in by_variant:
            continue
        title = "最後2問なし" if variant == "without_intention" else "最後2問あり"
        lines.append(f"## {title} ({variant})\n")
        for r in sorted(by_variant[variant], key=lambda x: x.dataset):
            lines.append(f"### {r.dataset}\n")
            lines.append(_format_row_for_md(r) + "\n")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _plot_diff_panels(
    rows: list[SummaryRow],
    diffs_by_variant_dataset: dict[tuple[str, str], np.ndarray],
    *,
    out_pdf: Path,
    out_png: Path,
) -> None:
    variants = ["without_intention", "with_intention"]
    datasets = sorted({r.dataset for r in rows})

    if not datasets:
        raise SystemExit("描画するデータがありません")

    fig, axes = plt.subplots(nrows=len(datasets), ncols=len(variants), figsize=(11.5, 3.8 * len(datasets)), sharey=True)
    if len(datasets) == 1 and len(variants) == 1:
        axes = np.array([[axes]])
    elif len(datasets) == 1:
        axes = np.array([axes])
    elif len(variants) == 1:
        axes = np.array([[a] for a in axes])

    y_values = []
    for (variant, dataset), diffs in diffs_by_variant_dataset.items():
        y_values.extend(list(diffs))
    ymin = float(np.min(y_values)) if y_values else -1.0
    ymax = float(np.max(y_values)) if y_values else 1.0
    pad = max(1.0, (ymax - ymin) * 0.12)
    ymin -= pad
    ymax += pad

    for i, dataset in enumerate(datasets):
        for j, variant in enumerate(variants):
            ax = axes[i, j]
            diffs = diffs_by_variant_dataset.get((variant, dataset), np.array([], dtype=float))
            ax.axhline(0.0, color="#444444", linewidth=1.0)

            if len(diffs) > 0:
                rng = np.random.default_rng(0)
                xs = rng.normal(loc=0.0, scale=0.04, size=len(diffs))
                ax.scatter(xs, diffs, s=45, color="#2a6fdb", zorder=3)

                summary = next((r for r in rows if r.variant == variant and r.dataset == dataset), None)
                if summary is not None:
                    ax.errorbar(
                        [0.0],
                        [summary.mean_diff],
                        yerr=[[summary.mean_diff - summary.mean_ci_low], [summary.mean_ci_high - summary.mean_diff]],
                        fmt="o",
                        color="#000000",
                        markersize=6,
                        capsize=4,
                        zorder=4,
                    )

            label_variant = "without intention" if variant == "without_intention" else "with intention"
            ax.set_title(f"{dataset} / {label_variant}")
            ax.set_xlim(-0.25, 0.25)
            ax.set_xticks([])
            ax.set_ylim(ymin, ymax)
            ax.grid(axis="y", color="#dddddd", linewidth=0.8)
            if j == 0:
                ax.set_ylabel("diff (AF - NF)")

    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="v2 合計点の差をまとめます（HCI向けの材料用）")
    parser.add_argument("--without-diff", default="data/questionnaire/derived/questionnaire_v2_total_diff.tsv")
    parser.add_argument("--with-diff", default="data/questionnaire/derived/questionnaire_v2_total_diff_with_intention.tsv")
    parser.add_argument("--out-tsv", default="data/questionnaire/derived/questionnaire_v2_stats.tsv")
    parser.add_argument("--out-md", default="data/questionnaire/derived/questionnaire_v2_stats.md")
    parser.add_argument("--out-fig-dir", default="paper_materials/figures/questionnaire")
    parser.add_argument("--n-boot", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    without_path = Path(args.without_diff)
    with_path = Path(args.with_diff)
    without_df = _load_diff_tsv(without_path)
    with_df = _load_diff_tsv(with_path)

    rows = []
    rows.extend(_summarize_variant(without_df, variant="without_intention", n_boot=args.n_boot, seed=args.seed))
    rows.extend(_summarize_variant(with_df, variant="with_intention", n_boot=args.n_boot, seed=args.seed))

    _write_tsv(Path(args.out_tsv), rows)
    _write_markdown(Path(args.out_md), rows, without_path=without_path, with_path=with_path)

    diffs_by = {}
    for variant, df in [("without_intention", without_df), ("with_intention", with_df)]:
        for dataset in sorted(df["dataset"].unique()):
            sub = df[df["dataset"] == dataset]["diff_system_minus_baseline"].astype(float).to_numpy()
            diffs_by[(variant, dataset)] = sub

    fig_dir = Path(args.out_fig_dir)
    _plot_diff_panels(
        rows,
        diffs_by,
        out_pdf=fig_dir / "questionnaire_v2_diff_panels.pdf",
        out_png=fig_dir / "questionnaire_v2_diff_panels.png",
    )

    print("OK")
    print(f"- {args.out_md}")
    print(f"- {args.out_tsv}")
    print(f"- {fig_dir / 'questionnaire_v2_diff_panels.pdf'}")


if __name__ == "__main__":
    main()
