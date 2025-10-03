"""IC score distribution analysis.

Summarises the distribution of Interface Complexity (IC) scores for either
tools or servers. It reads the artifacts emitted by ``analysis/ic_score.py``:

- Tools:  ``out/ic_tool.jsonl`` (JSON Lines)
- Servers: ``out/ic_server.json`` (JSON map keyed by server id)

Usage examples:
    # Tools (default)
    python -m analysis.ic_score_analysis

    # Servers
    python -m analysis.ic_score_analysis --scope server

    # Explicit path (supports .jsonl lines or a JSON object/array)
    python -m analysis.ic_score_analysis --scope tool --path out/ic_tool.jsonl

    # Emit a JSON summary as well
    python -m analysis.ic_score_analysis --as-json out/ic_score_summary.json

    # Save default JSON summaries under analysis/out
    python -m analysis.ic_score_analysis --save-json  # scope default is tools

    # Analyse both tools and servers and save JSON summaries
    python -m analysis.ic_score_analysis --scope both --save-json

    # Also print a text histogram and save a PNG figure
    python -m analysis.ic_score_analysis --hist --save-fig --bins 30
    # Or choose an explicit figure path
    python -m analysis.ic_score_analysis --save-fig --fig-path out/ic_score_hist_tool.png

    # Save a combined CDF (tools vs servers)
    python -m analysis.ic_score_analysis --save-cdf
    # Or to a custom path
    python -m analysis.ic_score_analysis --save-cdf --cdf-path out/ic_score_cdf_tools_vs_servers.png
    # Add custom threshold lines and include medians (default)
    python -m analysis.ic_score_analysis --save-cdf --vline 6 --vline 10
    # Include min/max markers (default) or disable with --no-extrema
    python -m analysis.ic_score_analysis --save-cdf --no-extrema
    # Control extrema marker size and labeling
    python -m analysis.ic_score_analysis --save-cdf --extrema-size 70
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import statistics as stats


# ------------------------------- Data types ---------------------------------


@dataclass
class ItemScore:
    """Generic score holder for either tools or servers."""

    id: str
    parent: Optional[str]
    name: str
    description: str
    score: float


@dataclass
class SummaryStats:
    count: int
    mean: float
    median: float
    min: float
    max: float
    variance: float
    stdev: float
    pvariance: float
    pstdev: float
    q1: float
    q2: float
    q3: float
    iqr: float
    p05: float
    p10: float
    p25: float
    p50: float
    p75: float
    p90: float
    p95: float
    p99: float
    range_: float
    geometric_mean: Optional[float]
    harmonic_mean: Optional[float]
    skewness: Optional[float]
    kurtosis_excess: Optional[float]


# ---------------------------- Helper functions ------------------------------


def _read_any_json_records(path: Path) -> List[Dict[str, Any]]:
    """Load records from either JSONL, JSON array, or JSON object mapping.

    Returns a list of dicts (raw records).
    """
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    # Heuristic: try JSONL first if there are multiple lines with braces
    if "\n" in text and text.lstrip().startswith("{"):
        # Could be a pretty JSON object; fall through to JSON
        pass
    if path.suffix.lower() == ".jsonl" or ("\n" in text and not text.lstrip().startswith("{")):
        records: List[Dict[str, Any]] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
        return records
    # Parse as a single JSON value
    data = json.loads(text)
    if isinstance(data, list):
        return [r for r in data if isinstance(r, dict)]
    if isinstance(data, dict):
        # If dict of records keyed by id
        if all(isinstance(v, dict) for v in data.values()):
            return list(data.values())
        # Or a single record
        return [data]
    return []


def _load_items(path: Path, scope: str) -> List[ItemScore]:
    records = _read_any_json_records(path)
    items: List[ItemScore] = []
    for rec in records:
        rec_scope = str(rec.get("scope") or "").lower()
        if rec_scope and rec_scope != scope:
            continue
        score = float(rec.get("score", 0.0))
        if scope == "tool":
            tool_id = str(rec.get("tool_id") or rec.get("id") or "")
            server_id = str(rec.get("server_id") or "")
            meta = rec.get("metadata") or {}
            name = str(meta.get("name") or "")
            description = str(meta.get("description") or "")
            items.append(ItemScore(id=tool_id, parent=server_id or None, name=name, description=description, score=score))
        else:  # server
            server_id = str(rec.get("server_id") or rec.get("id") or "")
            server_name = str(rec.get("server_name") or rec.get("name") or server_id)
            desc = ""
            items.append(ItemScore(id=server_id, parent=None, name=server_name, description=desc, score=score))
    if not items:
        raise SystemExit(f"No {scope} scores found in {path}")
    return items


def _compute_skew_kurtosis(values: Sequence[float]) -> Tuple[Optional[float], Optional[float]]:
    """Compute skewness and excess kurtosis using moment definitions.

    Returns (skewness, kurtosis_excess). For fewer than 3 values, returns None.
    """
    n = len(values)
    if n < 3:
        return None, None
    mean = stats.fmean(values)
    m2 = sum((x - mean) ** 2 for x in values) / n
    m3 = sum((x - mean) ** 3 for x in values) / n
    m4 = sum((x - mean) ** 4 for x in values) / n
    if m2 == 0.0:
        return 0.0, -3.0  # all equal: zero skew; kurtosis_excess of -3 (degenerate)
    skew = m3 / (m2 ** 1.5)
    kurtosis_excess = m4 / (m2 * m2) - 3.0
    return skew, kurtosis_excess


def _percentiles(values: Sequence[float], points: Sequence[int]) -> Dict[int, float]:
    """Compute requested integer percentiles using inclusive quantiles.

    Uses statistics.quantiles with ``n=100`` and method='inclusive' to obtain
    the 1..99 percentile cut points; 0 and 100 map to min and max respectively.
    """
    if not values:
        return {p: float("nan") for p in points}
    sorted_vals = sorted(values)
    # statistics.quantiles returns 99 cut points for n=100
    cuts = stats.quantiles(sorted_vals, n=100, method="inclusive")
    mapping: Dict[int, float] = {0: sorted_vals[0], 100: sorted_vals[-1]}
    for i, cut in enumerate(cuts, start=1):
        mapping[i] = cut
    return {p: mapping[max(0, min(100, int(p)))] for p in points}


def _histogram(values: Sequence[float], bins: int = 10) -> List[Tuple[float, float, int]]:
    """Simple fixed-width histogram over the value range.

    Returns a list of (start, end, count) for each bin.
    """
    if not values:
        return []
    lo, hi = min(values), max(values)
    if lo == hi:
        return [(lo, hi, len(values))]
    width = (hi - lo) / bins
    edges = [lo + i * width for i in range(bins)] + [hi]
    counts = [0 for _ in range(bins)]
    for v in values:
        if v == hi:
            idx = bins - 1
        else:
            idx = int((v - lo) / width)
            idx = max(0, min(bins - 1, idx))
        counts[idx] += 1
    return [(edges[i], edges[i + 1], counts[i]) for i in range(bins)]


def compute_summary(values: Sequence[float]) -> SummaryStats:
    if not values:
        raise ValueError("compute_summary requires at least one value")
    vals = list(values)
    vals.sort()
    count = len(vals)
    mean = stats.fmean(vals)
    median = stats.median(vals)
    vmin, vmax = vals[0], vals[-1]
    # Sample and population moments
    variance = stats.variance(vals) if count > 1 else 0.0
    stdev = stats.stdev(vals) if count > 1 else 0.0
    pvariance = stats.pvariance(vals) if count > 1 else 0.0
    pstdev = stats.pstdev(vals) if count > 1 else 0.0
    # Quartiles (inclusive method approximates Excel's percentiles)
    q1, q2, q3 = stats.quantiles(vals, n=4, method="inclusive")
    iqr = q3 - q1
    percent = _percentiles(vals, [5, 10, 25, 50, 75, 90, 95, 99])
    # Means that may be undefined for non-positive values
    geometric_mean = None
    harmonic_mean = None
    if all(v > 0 for v in vals):
        try:
            geometric_mean = stats.geometric_mean(vals)
        except Exception:
            geometric_mean = None
        try:
            harmonic_mean = stats.harmonic_mean(vals)
        except Exception:
            harmonic_mean = None
    skew, kurt_ex = _compute_skew_kurtosis(vals)
    return SummaryStats(
        count=count,
        mean=mean,
        median=median,
        min=vmin,
        max=vmax,
        variance=variance,
        stdev=stdev,
        pvariance=pvariance,
        pstdev=pstdev,
        q1=q1,
        q2=q2,
        q3=q3,
        iqr=iqr,
        p05=percent[5],
        p10=percent[10],
        p25=percent[25],
        p50=percent[50],
        p75=percent[75],
        p90=percent[90],
        p95=percent[95],
        p99=percent[99],
        range_=vmax - vmin,
        geometric_mean=geometric_mean,
        harmonic_mean=harmonic_mean,
        skewness=skew,
        kurtosis_excess=kurt_ex,
    )


def _format_float(x: Optional[float]) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "-"
    return f"{x:.6f}"


def print_summary(stats_: SummaryStats, label: str = "tools") -> None:
    title = f"IC Score Distribution ({label})"
    print(title)
    print("-" * 32)
    print(f"Count: {stats_.count}")
    print(f"Mean:  {_format_float(stats_.mean)}")
    print(f"Median:{_format_float(stats_.median)}")
    print(f"Min:   {_format_float(stats_.min)}")
    print(f"Max:   {_format_float(stats_.max)}")
    print(f"Range: {_format_float(stats_.range_)}")
    print()
    print("Spread (variance/stdev)")
    print(f"  Variance:       {_format_float(stats_.variance)} (sample)")
    print(f"  Std Dev:        {_format_float(stats_.stdev)} (sample)")
    print(f"  PVariance:      {_format_float(stats_.pvariance)} (population)")
    print(f"  PStd Dev:       {_format_float(stats_.pstdev)} (population)")
    print()
    print("Quartiles / Percentiles")
    print(f"  Q1:  {_format_float(stats_.q1)}  Q2/Median: {_format_float(stats_.q2)}  Q3: {_format_float(stats_.q3)}  IQR: {_format_float(stats_.iqr)}")
    print(f"  p05: {_format_float(stats_.p05)}  p10: {_format_float(stats_.p10)}  p25: {_format_float(stats_.p25)}  p50: {_format_float(stats_.p50)}")
    print(f"  p75: {_format_float(stats_.p75)}  p90: {_format_float(stats_.p90)}  p95: {_format_float(stats_.p95)}  p99: {_format_float(stats_.p99)}")
    print()
    print("Other Means / Moments")
    print(f"  Geometric Mean: {_format_float(stats_.geometric_mean)}")
    print(f"  Harmonic Mean:  {_format_float(stats_.harmonic_mean)}")
    print(f"  Skewness:       {_format_float(stats_.skewness)}")
    print(f"  Kurtosis (ex):  {_format_float(stats_.kurtosis_excess)}")


def _unused_histogram(values: Sequence[float], bins: int = 10) -> List[Tuple[float, float, int]]:
    # Retained for potential future use; not printed by default.
    return _histogram(values, bins=bins)


def print_ascii_histogram(values: Sequence[float], bins: int = 20, width: int = 50) -> None:
    """Print a simple ASCII histogram to stdout.

    Uses fixed-width bins across the data range. The bar length is scaled to
    ``width`` characters based on the max bin count.
    """
    buckets = _histogram(values, bins=bins)
    if not buckets:
        print("No data to plot histogram.")
        return
    max_count = max(c for _, _, c in buckets) or 1
    print("\nHistogram (ASCII)")
    print("-" * 18)
    for start, end, count in buckets:
        bar_len = int(round((count / max_count) * width))
        bar = "#" * bar_len
        print(f"[{start:>7.3f}, {end:>7.3f}) | {count:>4d} {bar}")


def save_histogram_figure(values: Sequence[float], bins: int, out_path: Path, title: str) -> None:
    """Save a histogram figure to ``out_path`` using matplotlib if available.

    If matplotlib is not installed, a clear error is raised to guide the user
    to install it. This function does not show the figure; it only saves it.
    """
    try:
        # Use a headless backend to avoid GUI requirements in sandboxed envs
        import matplotlib  # type: ignore
        try:  # pragma: no cover - backend selection is environment-specific
            matplotlib.use("Agg", force=True)
        except Exception:
            pass
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "matplotlib is required to save figures. Install with 'pip install matplotlib'."
        ) from exc

    fig = plt.figure(figsize=(8, 4.5), dpi=120)
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(values, bins=bins, color="#4C78A8", edgecolor="white", alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel("IC Score")
    ax.set_ylabel("Count")
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def _ecdf(values: Sequence[float]) -> Tuple[List[float], List[float]]:
    """Compute the empirical CDF (x, y) for given values.

    Returns sorted x values and cumulative probabilities in [0,1].
    """
    if not values:
        return [], []
    xs = sorted(values)
    n = len(xs)
    ys = [(i + 1) / n for i in range(n)]
    return xs, ys


def save_cdf_figure(
    tool_scores: Sequence[float],
    server_scores: Sequence[float],
    out_path: Path,
    thresholds: Optional[Sequence[float]] = None,
    show_medians: bool = True,
    show_extrema: bool = True,
    extrema_marker_size: float = 60.0,
    label_extrema: bool = True,
) -> None:
    """Save a combined CDF figure for tools and servers.

    Uses a headless matplotlib backend. The y-axis spans [0,1].
    """
    try:
        import matplotlib  # type: ignore
        try:  # pragma: no cover
            matplotlib.use("Agg", force=True)
        except Exception:
            pass
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            "matplotlib is required to save figures. Install with 'pip install matplotlib'."
        ) from exc

    tx, ty = _ecdf(tool_scores)
    sx, sy = _ecdf(server_scores)
    if not tx or not sx:
        raise SystemExit("Insufficient data to plot CDF (missing tool or server scores)")

    fig = plt.figure(figsize=(8, 4.5), dpi=120)
    ax = fig.add_subplot(1, 1, 1)
    ax.step(tx, ty, where="post", label="Tools", color="#4C78A8")
    ax.step(sx, sy, where="post", label="Servers", color="#F58518")
    # Optional medians
    if show_medians:
        try:
            t_med = stats.median(tool_scores)
            s_med = stats.median(server_scores)
        except Exception:
            t_med = None
            s_med = None
        if t_med is not None:
            ax.axvline(t_med, color="#4C78A8", linestyle="--", alpha=0.9, label=f"Tool median: {t_med:.2f}")
        if s_med is not None:
            ax.axvline(s_med, color="#F58518", linestyle="--", alpha=0.9, label=f"Server median: {s_med:.2f}")
    # Optional extrema (min/max) — markers only, no vertical lines
    if show_extrema:
        try:
            t_min, t_max = min(tool_scores), max(tool_scores)
            s_min, s_max = min(server_scores), max(server_scores)
            tn = max(1, len(tool_scores))
            sn = max(1, len(server_scores))
        except Exception:
            t_min = t_max = s_min = s_max = None
        # Use scatter for consistent sizing; add a subtle edge for contrast
        edge_color = "#222222"
        ms = float(extrema_marker_size)
        def _clamp01(y: float) -> float:
            return max(0.0, min(1.0, y))
        if t_min is not None and t_max is not None:
            ax.scatter([t_min], [_clamp01(1.0/tn)], s=ms, color="#4C78A8", edgecolors=edge_color, linewidths=0.5, zorder=5)
            ax.scatter([t_max], [1.0], s=ms, color="#4C78A8", edgecolors=edge_color, linewidths=0.5, zorder=5)
            if label_extrema:
                # Use offset annotations for readability (no legend entries)
                ax.annotate(
                    f"Tool min {t_min:.2f}",
                    xy=(t_min, _clamp01(1.0/tn)),
                    xytext=(-14, 10), textcoords="offset points",
                    color="#4C78A8", fontsize=9, ha="right", va="bottom",
                    arrowprops=dict(arrowstyle="-", color="#4C78A8", lw=0.6, alpha=0.8),
                )
                ax.annotate(
                    f"Tool max {t_max:.2f}",
                    xy=(t_max, 1.0),
                    xytext=(-14, -12), textcoords="offset points",
                    color="#4C78A8", fontsize=9, ha="right", va="top",
                    arrowprops=dict(arrowstyle="-", color="#4C78A8", lw=0.6, alpha=0.8),
                )
        if s_min is not None and s_max is not None:
            ax.scatter([s_min], [_clamp01(1.0/sn)], s=ms, color="#F58518", edgecolors=edge_color, linewidths=0.5, zorder=5)
            ax.scatter([s_max], [1.0], s=ms, color="#F58518", edgecolors=edge_color, linewidths=0.5, zorder=5)
            if label_extrema:
                ax.annotate(
                    f"Server min {s_min:.2f}",
                    xy=(s_min, _clamp01(1.0/sn)),
                    xytext=(14, 10), textcoords="offset points",
                    color="#F58518", fontsize=9, ha="left", va="bottom",
                    arrowprops=dict(arrowstyle="-", color="#F58518", lw=0.6, alpha=0.8),
                )
                ax.annotate(
                    f"Server max {s_max:.2f}",
                    xy=(s_max, 1.0),
                    xytext=(14, -12), textcoords="offset points",
                    color="#F58518", fontsize=9, ha="left", va="top",
                    arrowprops=dict(arrowstyle="-", color="#F58518", lw=0.6, alpha=0.8),
                )
        # No legend entries for extrema to avoid clutter; labels are drawn as text
    # Optional custom thresholds
    if thresholds:
        for x in thresholds:
            try:
                xv = float(x)
            except Exception:
                continue
            ax.axvline(xv, color="#6C6C6C", linestyle=":", alpha=0.8, label=f"Threshold: {xv:.2f}")
    ax.set_title("IC Score CDF (Tools vs Servers)")
    ax.set_xlabel("IC Score")
    ax.set_ylabel("Cumulative Probability")
    ax.set_ylim(0.0, 1.0)
    # Align x limits to combined range
    xmin = min(tx[0], sx[0])
    xmax = max(tx[-1], sx[-1])
    ax.set_xlim(xmin, xmax)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def to_json_dict(summary: SummaryStats) -> Dict[str, float]:
    return {
        "count": summary.count,
        "mean": summary.mean,
        "median": summary.median,
        "min": summary.min,
        "max": summary.max,
        "range": summary.range_,
        "variance": summary.variance,
        "stdev": summary.stdev,
        "pvariance": summary.pvariance,
        "pstdev": summary.pstdev,
        "q1": summary.q1,
        "q2": summary.q2,
        "q3": summary.q3,
        "iqr": summary.iqr,
        "p05": summary.p05,
        "p10": summary.p10,
        "p25": summary.p25,
        "p50": summary.p50,
        "p75": summary.p75,
        "p90": summary.p90,
        "p95": summary.p95,
        "p99": summary.p99,
        "geometric_mean": summary.geometric_mean,
        "harmonic_mean": summary.harmonic_mean,
        "skewness": summary.skewness,
        "kurtosis_excess": summary.kurtosis_excess,
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Analyse IC score distribution for tools or servers")
    parser.add_argument("--scope", choices=["tool", "server", "both"], default="tool", help="Which score file to analyse")
    parser.add_argument("--path", type=Path, help="Path to score file (.jsonl or .json)")
    # Backward-compat alias for tools
    parser.add_argument("--ic-tool-path", type=Path, help="Deprecated: path to ic_tool.jsonl")
    parser.add_argument("--as-json", type=Path, help="Optional path to write summary JSON")
    parser.add_argument("--save-json", action="store_true", help="Write summary JSON(s) to analysis/out for selected scope")
    parser.add_argument("--tool-path", type=Path, help="Override path for tool scores (when --scope both)")
    parser.add_argument("--server-path", type=Path, help="Override path for server scores (when --scope both)")
    # Histogram controls
    parser.add_argument("--hist", action="store_true", help="Print an ASCII histogram of the scores")
    parser.add_argument("--bins", type=int, default=20, help="Number of histogram bins")
    parser.add_argument("--save-fig", action="store_true", help="Save a PNG histogram figure (requires matplotlib)")
    parser.add_argument("--fig-path", type=Path, help="Optional figure output path (defaults under analysis/out)")
    # CDF combined figure
    parser.add_argument("--save-cdf", action="store_true", help="Save a combined CDF figure for tools and servers")
    parser.add_argument("--cdf-path", type=Path, help="Optional CDF output path (defaults under analysis/out)")
    parser.add_argument("--vline", type=float, action="append", help="Add vertical threshold line(s) to the CDF; repeat for multiple")
    parser.add_argument("--no-medians", action="store_true", help="Do not draw median lines on the CDF")
    parser.add_argument("--no-extrema", action="store_true", help="Do not draw min/max markers on the CDF")
    parser.add_argument("--extrema-size", type=float, default=60.0, help="Marker area for extrema points (matplotlib 's' for scatter)")
    parser.add_argument("--no-extrema-labels", action="store_true", help="Do not draw text labels next to extrema markers")
    args = parser.parse_args(argv)

    default_tool = Path(__file__).parent / "out" / "ic_tool.jsonl"
    default_server = Path(__file__).parent / "out" / "ic_server.json"

    # Resolve per-scope paths
    tool_path = args.tool_path or args.ic_tool_path or (args.path if args.path and args.scope == "tool" else None) or default_tool
    server_path = args.server_path or (args.path if args.path and args.scope == "server" else None) or default_server

    def analyse_one(scope: str, in_path: Path) -> Tuple[SummaryStats, List[float]]:
        if not in_path.exists():
            raise SystemExit(f"File not found: {in_path}")
        items = _load_items(in_path, scope=scope)
        scores = [it.score for it in items]
        summary = compute_summary(scores)
        label = "tools" if scope == "tool" else "servers"
        print_summary(summary, label=label)
        return summary, scores

    if args.scope == "both":
        # Tools
        t_summary, t_scores = analyse_one("tool", tool_path)
        if args.hist:
            print_ascii_histogram(t_scores, bins=max(1, int(args.bins)))
        if args.save_fig:
            default_fig_t = (Path(__file__).parent / "out" / "ic_score_hist_tool.png")
            save_histogram_figure(t_scores, bins=max(1, int(args.bins)), out_path=args.fig_path or default_fig_t, title="IC Score Distribution (tools)")
            print(f"\nSaved histogram figure: {args.fig_path or default_fig_t}")
        # Servers
        print()
        s_summary, s_scores = analyse_one("server", server_path)
        if args.hist:
            print_ascii_histogram(s_scores, bins=max(1, int(args.bins)))
        if args.save_fig:
            default_fig_s = (Path(__file__).parent / "out" / "ic_score_hist_server.png")
            save_histogram_figure(s_scores, bins=max(1, int(args.bins)), out_path=args.fig_path or default_fig_s, title="IC Score Distribution (servers)")
            print(f"\nSaved histogram figure: {args.fig_path or default_fig_s}")

        if args.save_cdf:
            default_cdf = Path(__file__).parent / "out" / "ic_score_cdf_tools_vs_servers.png"
            cdf_path = args.cdf_path or default_cdf
            save_cdf_figure(
                t_scores,
                s_scores,
                out_path=cdf_path,
                thresholds=args.vline,
                show_medians=(not args.no_medians),
                show_extrema=(not args.no_extrema),
                extrema_marker_size=args.extrema_size,
                label_extrema=(not args.no_extrema_labels),
            )
            print(f"\nSaved combined CDF: {cdf_path}")

        if args.as_json:
            raise SystemExit("--as-json is not supported when --scope both; use --save-json instead")
        if args.save_json:
            out_dir = Path(__file__).parent / "out"
            out_dir.mkdir(parents=True, exist_ok=True)
            t_path = out_dir / "ic_score_summary_tool.json"
            s_path = out_dir / "ic_score_summary_server.json"
            with t_path.open("w", encoding="utf-8") as fh:
                json.dump(to_json_dict(t_summary), fh, ensure_ascii=False, indent=2)
            with s_path.open("w", encoding="utf-8") as fh:
                json.dump(to_json_dict(s_summary), fh, ensure_ascii=False, indent=2)
            print(f"\nWrote JSON summaries: {t_path} , {s_path}")
        return

    # Single-scope flow
    in_path: Path = tool_path if args.scope == "tool" else server_path
    if not in_path.exists():
        raise SystemExit(f"File not found: {in_path}")

    items = _load_items(in_path, scope=args.scope)
    scores = [it.score for it in items]
    summary = compute_summary(scores)
    label = "tools" if args.scope == "tool" else "servers"
    print_summary(summary, label=label)

    if args.hist:
        print_ascii_histogram(scores, bins=max(1, int(args.bins)))

    if args.save_fig:
        default_fig = (Path(__file__).parent / "out" / f"ic_score_hist_{args.scope}.png")
        fig_path = args.fig_path or default_fig
        title = f"IC Score Distribution ({label})"
        save_histogram_figure(scores, bins=max(1, int(args.bins)), out_path=fig_path, title=title)
        print(f"\nSaved histogram figure: {fig_path}")

    if args.save_cdf:
        # Ensure both sets are available
        if args.scope == "tool":
            other_scores = [it.score for it in _load_items(server_path, scope="server")]
            t_scores, s_scores = scores, other_scores
        elif args.scope == "server":
            other_scores = [it.score for it in _load_items(tool_path, scope="tool")]
            t_scores, s_scores = other_scores, scores
        else:
            # Should not hit here; handled in 'both' branch
            other_scores = [it.score for it in _load_items(server_path, scope="server")]
            t_scores, s_scores = scores, other_scores
        default_cdf = Path(__file__).parent / "out" / "ic_score_cdf_tools_vs_servers.png"
        cdf_path = args.cdf_path or default_cdf
        save_cdf_figure(
            t_scores,
            s_scores,
            out_path=cdf_path,
            thresholds=args.vline,
            show_medians=(not args.no_medians),
            show_extrema=(not args.no_extrema),
            extrema_marker_size=args.extrema_size,
            label_extrema=(not args.no_extrema_labels),
        )
        print(f"\nSaved combined CDF: {cdf_path}")

    if args.as_json:
        payload = to_json_dict(summary)
        args.as_json.parent.mkdir(parents=True, exist_ok=True)
        with args.as_json.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        print(f"\nWrote JSON summary: {args.as_json}")
    if args.save_json:
        default_json = Path(__file__).parent / "out" / f"ic_score_summary_{args.scope}.json"
        default_json.parent.mkdir(parents=True, exist_ok=True)
        with default_json.open("w", encoding="utf-8") as fh:
            json.dump(to_json_dict(summary), fh, ensure_ascii=False, indent=2)
        print(f"\nWrote JSON summary: {default_json}")


if __name__ == "__main__":
    main()
