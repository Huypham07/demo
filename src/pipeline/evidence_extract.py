from typing import Optional

import pandas as pd

from src.pipeline.evidence_detector import process_dataframe as detect_evidence
from src.pipeline.evidence_linker import run_linking_variant


def evidence_extract(
    df: pd.DataFrame,
    variant: str = "nli",
    config: Optional[dict] = None,
    corpus_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    if "text" not in df.columns and "sentence" in df.columns:
        df = df.copy()
        df["text"] = df["sentence"]

    df = detect_evidence(df)

    if corpus_df is not None:
        if "text" not in corpus_df.columns and "sentence" in corpus_df.columns:
            corpus_df = corpus_df.copy()
            corpus_df["text"] = corpus_df["sentence"]
        corpus_df = detect_evidence(corpus_df)

    links_df = run_linking_variant(
        df, variant=variant, text_column="text", config=config, corpus_df=corpus_df
    )

    df["has_evidence"] = links_df["evidence_found"].values
    df["best_evidence"] = links_df["best_evidence"].values
    df["similarity_score"] = links_df["similarity_score"].values
    df["num_evidence"] = links_df["num_evidence"].values
    df["search_method"] = links_df["search_method"].values
    df["nli_entailment_score"] = links_df["nli_entailment_score"].values
    df["nli_label"] = links_df["nli_label"].values
    df["evidence_variant"] = variant
    return df
