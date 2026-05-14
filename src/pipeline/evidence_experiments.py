from pathlib import Path
import argparse
from typing import Optional
import pandas as pd

from src.pipeline.evidence_detector import process_dataframe as detect_evidence_batch
from src.pipeline.evidence_linker import EVIDENCE_VARIANTS, run_linking_variant

def _prepare_base_df(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    if "text" not in work.columns and "sentence" in work.columns:
        work["text"] = work["sentence"]
    return detect_evidence_batch(work)

def apply_evidence_variant(
    df: pd.DataFrame, variant: str, config: Optional[dict] = None
) -> pd.DataFrame:
    base_df = _prepare_base_df(df)

    links_df = run_linking_variant(
        base_df, variant=variant, text_column="text", config=config
    )

    base_df = base_df.copy()
    base_df["best_evidence"] = links_df["best_evidence"].values
    base_df["similarity_score"] = links_df["similarity_score"].values
    base_df["num_evidence"] = links_df["num_evidence"].values
    base_df["search_method"] = links_df["search_method"].values
    base_df["nli_entailment_score"] = links_df["nli_entailment_score"].values
    base_df["nli_label"] = links_df["nli_label"].values
    base_df["evidence_variant"] = variant

    return base_df

def summarize_variant(df: pd.DataFrame, variant: str) -> dict:
    total = len(df)
    evidence_rate = float(df["has_evidence"].mean()) if total else 0.0
    avg_similarity = float(df.get("similarity_score", pd.Series([0.0] * max(total, 1))).mean())
    avg_nli_entailment = float(df["nli_entailment_score"].mean()) if "nli_entailment_score" in df.columns and total else 0.0

    action_col = "action_label" if "action_label" in df.columns else None

    high_risk_rate = 0.0
    if action_col is not None and total:
        mask = (df[action_col] == "Indeterminate") & (~df["has_evidence"].astype(bool))
        high_risk_rate = float(mask.mean())

    nli_dist = {}
    if "nli_label" in df.columns and total:
        nli_dist = df["nli_label"].value_counts(normalize=True).to_dict()

    return {
        "variant": variant,
        "rows": total,
        "evidence_rate": round(evidence_rate, 4),
        "avg_similarity": round(avg_similarity, 4),
        "avg_nli_entailment": round(avg_nli_entailment, 4),
        "nli_entailment_pct": round(nli_dist.get("entailment", 0.0), 4),
        "nli_contradiction_pct": round(nli_dist.get("contradiction", 0.0), 4),
        "high_risk_rate": round(high_risk_rate, 4),
    }

def run_experiments(
    input_path: str,
    output_dir: str,
    variants: Optional[list] = None,
    config: Optional[dict] = None,
) -> pd.DataFrame:
    input_file = Path(input_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    df = pd.read_parquet(input_file)

    all_variants = ["nli", "window", "no_nli"]
    selected = variants if variants else all_variants

    records = []
    for variant in selected:
        print(f"\n[Evidence Experiment] Running variant: {variant}")
        result_df = apply_evidence_variant(df, variant=variant, config=config)

        output_file = out_dir / f"evidence_{variant}.parquet"
        result_df.to_parquet(output_file, index=False)
        print(f"Saved: {output_file}")

        records.append(summarize_variant(result_df, variant))

    summary_df = pd.DataFrame(records).sort_values("evidence_rate", ascending=False)
    summary_file = out_dir / "evidence_variant_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"\nSaved summary: {summary_file}")

    return summary_df

def main():
    parser = argparse.ArgumentParser(description="Run evidence variants for ablation experiments")
    parser.add_argument(
        "--input",
        type=str,
        default="data/corpus/actionability_sentences.parquet",
        help="Input parquet file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/experiments/evidence",
        help="Output directory",
    )
    parser.add_argument(
        "--variants",
        type=str,
        default=None,
        help="Comma-separated variants. Example: nli,window,no_nli",
    )

    args = parser.parse_args()
    variants = [v.strip() for v in args.variants.split(",")] if args.variants else None

    summary_df = run_experiments(args.input, args.output_dir, variants=variants)
    print("\n=== Evidence Variant Summary ===")
    print(summary_df.to_string(index=False))

if __name__ == "__main__":
    main()
