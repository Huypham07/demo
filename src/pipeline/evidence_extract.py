from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

from src.pipeline.evidence_detector import process_dataframe as detect_evidence
from src.pipeline.evidence_linker import run_linking_variant


def evidence_extract(
    df: pd.DataFrame,
    variant: str = "nli",
    config: Optional[dict] = None,
    corpus_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    df = df.copy()
    if "text" not in df.columns and "sentence" in df.columns:
        df["text"] = df["sentence"]

    if corpus_df is not None:
        corpus_df = corpus_df.copy()
        if "text" not in corpus_df.columns and "sentence" in corpus_df.columns:
            corpus_df["text"] = corpus_df["sentence"]

        corpus_df = detect_evidence(corpus_df)

        ev_lookup = corpus_df.set_index("sent_id")[["evidence_types", "kpi_values"]]
        df["evidence_types"] = df["sent_id"].map(ev_lookup["evidence_types"])
        df["kpi_values"] = df["sent_id"].map(ev_lookup["kpi_values"])
        df["has_evidence"] = df["evidence_types"].apply(
            lambda x: bool(x) if isinstance(x, list) else False
        )
    else:
        df = detect_evidence(df)

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


def main() -> None:
    with open("config/pipeline.yml") as f:
        cfg = yaml.safe_load(f)

    ACTION_PATH = Path("data/corpus/actionability_sentences.parquet")
    CORPUS_PATH = Path("data/corpus/sentences.parquet")
    if not ACTION_PATH.exists():
        raise FileNotFoundError(
            f"{ACTION_PATH} không tồn tại.\n"
            "Chạy phần Phụ lục ở cuối notebook trước."
        )

    df_corpus = pd.read_parquet(CORPUS_PATH)
    df = pd.read_parquet(ACTION_PATH)

    print(f"Corpus: {len(df):,} ESG sentences")

    OUT_EV = Path("outputs/experiments/evidence")
    OUT_EV.mkdir(parents=True, exist_ok=True)
    rq2 = {}
    for variant in ["nli", "window", "no_nli"]:
        cache = OUT_EV / f"evidence_{variant}.parquet"
        if cache.exists():
            df_v = pd.read_parquet(cache)
            print(f"[{variant}] loaded from cache")
        else:
            print(f"[{variant}] computing…")
            df_v = evidence_extract(df.copy(), variant=variant, config=cfg, corpus_df=df_corpus)
            df_v.to_parquet(cache, index=False)

        n_total = len(df_v)
        n_ev = int(df_v["has_evidence"].sum()) if "has_evidence" in df_v.columns else 0
        avg_sim = float(df_v["similarity_score"].mean()) if "similarity_score" in df_v.columns else 0.0
        nli_cnt = (
            df_v["nli_label"].value_counts(dropna=False).to_dict()
            if "nli_label" in df_v.columns else {}
        )
        rq2[variant] = dict(
            evidence_rate_pct=round(n_ev / n_total * 100, 1),
            avg_similarity=round(avg_sim, 4),
            entailment_pct=round(nli_cnt.get("entailment", 0) / max(n_ev, 1) * 100, 1),
            contradiction_pct=round(nli_cnt.get("contradiction", 0) / max(n_ev, 1) * 100, 1),
        )


if __name__ == "__main__":
    main()
