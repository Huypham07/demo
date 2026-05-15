"""
Compute evidence type distribution statistics for thesis Chapter 4.

Usage:
    python -m scripts.compute_evidence_stats

Outputs:
    1. Distribution across full corpus   (data/corpus/sentences.parquet)
    2. Distribution across ESG sentences (data/corpus/esg_sentences.parquet)
    3. Distribution for NLI experiment claims, split by linked-evidence flag
       (outputs/experiments/evidence/evidence_nli.parquet)
    4. LaTeX-ready table printed to stdout — copy into c4_chapter.tex
"""

import sys
from itertools import combinations
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.evidence_detector import EVIDENCE_TYPES, detect_evidence


# ── helpers ───────────────────────────────────────────────────────────────────

def _tag_series(series: pd.Series) -> pd.DataFrame:
    """
    Run detect_evidence on every sentence in a Series.
    Returns a DataFrame with one bool column per evidence type plus 'has_any'.
    """
    results = series.apply(lambda t: detect_evidence(str(t)))
    typed = {
        etype: results.apply(lambda r: etype in r["evidence_types"])
        for etype in EVIDENCE_TYPES
    }
    typed["has_any"] = results.apply(lambda r: r["has_evidence"])
    return pd.DataFrame(typed, index=series.index)


def _tag_from_lists(series: pd.Series) -> pd.DataFrame:
    """Build bool columns from an already-computed evidence_types list column."""
    typed = {
        etype: series.apply(lambda lst: etype in lst if isinstance(lst, list) else False)
        for etype in EVIDENCE_TYPES
    }
    typed["has_any"] = pd.DataFrame(typed).any(axis=1)
    return pd.DataFrame(typed, index=series.index)


def _dist_table(typed: pd.DataFrame, total: int, title: str) -> None:
    sep = "=" * 62
    print(f"\n{sep}")
    print(f"  {title}")
    print(f"  n = {total:,}")
    print(sep)
    print(f"  {'Loại bằng chứng':<20} {'Số câu':>8}  {'Tỉ lệ':>8}")
    print(f"  {'-'*40}")
    for etype in EVIDENCE_TYPES:
        if etype not in typed.columns:
            continue
        n = int(typed[etype].sum())
        print(f"  {etype:<20} {n:>8,}  {n / total * 100:>7.1f}%")
    print(f"  {'-'*40}")
    n_any = int(typed["has_any"].sum())
    n_none = total - n_any
    print(f"  {'Ít nhất một loại':<20} {n_any:>8,}  {n_any / total * 100:>7.1f}%")
    print(f"  {'Không có loại nào':<20} {n_none:>8,}  {n_none / total * 100:>7.1f}%")


def _cooccurrence(typed: pd.DataFrame, total: int) -> None:
    print("\n  Co-occurrence (top pairs):")
    pairs = []
    for a, b in combinations(EVIDENCE_TYPES.keys(), 2):
        if a in typed.columns and b in typed.columns:
            n = int((typed[a] & typed[b]).sum())
            if n > 0:
                pairs.append((n, a, b))
    for n, a, b in sorted(pairs, reverse=True)[:6]:
        print(f"    {a} ∩ {b}: {n:,}  ({n / total * 100:.1f}%)")


def _latex_table(typed: pd.DataFrame, total: int) -> None:
    """Print the numbers formatted for direct paste into the thesis LaTeX table."""
    print("\n" + "=" * 62)
    print("  LATEX TABLE — paste into c4_chapter.tex")
    print("=" * 62)
    for etype in EVIDENCE_TYPES:
        if etype not in typed.columns:
            continue
        n = int(typed[etype].sum())
        pct = n / total * 100
        print(f"  {etype:<12} & {n:,} & {pct:.1f}\\% \\\\")
    print("  \\midrule")
    n_any = int(typed["has_any"].sum())
    print(f"  {'Ít nhất một loại':<12} & {n_any:,} & {n_any / total * 100:.1f}\\% \\\\")
    print("=" * 62)


# ── main sections ─────────────────────────────────────────────────────────────

def analyse_full_corpus(path: Path) -> pd.DataFrame:
    print(f"\nLoading full corpus: {path}")
    df = pd.read_parquet(path)
    print(f"  {len(df):,} sentences")

    typed = _tag_series(df["sentence"])
    _dist_table(typed, len(df), "Full corpus (sentences.parquet)")
    _cooccurrence(typed, len(df))
    _latex_table(typed, len(df))
    return typed


def analyse_esg_sentences(path: Path) -> pd.DataFrame:
    print(f"\nLoading ESG sentences: {path}")
    df = pd.read_parquet(path)
    print(f"  {len(df):,} ESG sentences")

    text_col = "sentence" if "sentence" in df.columns else "text"
    typed = _tag_series(df[text_col])
    _dist_table(typed, len(df), "ESG sentences (esg_sentences.parquet)")
    _cooccurrence(typed, len(df))
    return typed


def analyse_nli_experiment(path: Path) -> None:
    print(f"\nLoading NLI experiment: {path}")
    df = pd.read_parquet(path)
    print(f"  {len(df):,} ESG claims")
    print(f"  Columns: {list(df.columns)}")

    # Resolve evidence flag column
    ev_col = next(
        (c for c in ("has_evidence", "evidence_found") if c in df.columns), None
    )
    if ev_col is None:
        print("  No evidence flag column found — skipping NLI breakdown.")
        return

    # Resolve text column
    text_col = next(
        (c for c in ("sentence", "text") if c in df.columns), None
    )
    if text_col is None:
        print("  No text column found — skipping NLI breakdown.")
        return

    # Build type indicators
    if "evidence_types" in df.columns:
        typed = _tag_from_lists(df["evidence_types"])
    else:
        typed = _tag_series(df[text_col])

    with_ev = df[ev_col].astype(bool)

    _dist_table(typed, len(df), "NLI experiment — ALL ESG claims")
    _dist_table(typed[with_ev], int(with_ev.sum()), "NLI experiment — WITH linked evidence")
    _dist_table(typed[~with_ev], int((~with_ev).sum()), "NLI experiment — WITHOUT linked evidence")


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    sentences_path = Path("data/corpus/sentences.parquet")
    esg_path = Path("data/corpus/esg_sentences.parquet")
    nli_path = Path("outputs/experiments/evidence/evidence_nli.parquet")

    if sentences_path.exists():
        analyse_full_corpus(sentences_path)
    else:
        print(f"Not found: {sentences_path}")

    if esg_path.exists():
        analyse_esg_sentences(esg_path)
    else:
        print(f"Not found: {esg_path}")

    if nli_path.exists():
        analyse_nli_experiment(nli_path)
    else:
        print(f"Not found: {nli_path} — run evidence_experiments first.")

    print("\nDone. Copy LaTeX numbers above into thesis/chapters/c4/c4_chapter.tex.")


if __name__ == "__main__":
    main()
