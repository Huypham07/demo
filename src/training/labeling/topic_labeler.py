import numpy as np
import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.labeling.grounded_rules import match_topic_grounded

SENTENCES_PATH = Path("data/corpus/sentences.parquet")
LLM_LABELS_PATH = Path("data/labels/topic/llm_prelabels.parquet")
OUTPUT_PATH = Path("data/labels/topic")

TOPICS = ["E", "S_labor", "S_community", "S_product", "G", "Non_ESG"]
TOPIC_TO_IDX = {t: i for i, t in enumerate(TOPICS)}

def create_hybrid_labels(min_confidence: float = 0.5) -> pd.DataFrame:
    print("Loading sentences...")
    df = pd.read_parquet(SENTENCES_PATH)
    print(f"Total sentences: {len(df):,}")
    
    print("Loading LLM labels...")
    if not LLM_LABELS_PATH.exists():
        print(f"ERROR: LLM labels not found at {LLM_LABELS_PATH}")
        print("Run: python -m src.topic_classification.llm_labeler")
        return None
    
    llm = pd.read_parquet(LLM_LABELS_PATH)
    print(f"LLM labels: {len(llm):,}")
    
    df = df.merge(
        llm[["sent_id", "llm_label", "llm_confidence", "llm_reason"]],
        on="sent_id", how="left"
    )
    
    has_llm = df["llm_label"].notna()
    print(f"LLM coverage: {has_llm.sum():,} ({has_llm.mean()*100:.1f}%)")
    
    print("\nApplying GRI rules...")
    rule_results = df.apply(
        lambda row: match_topic_grounded(
            str(row.get("sentence", "")),
            str(row.get("section_title", ""))
        ),
        axis=1
    )
    
    df["rule_label"] = rule_results.apply(lambda x: x[0] if x else None)
    df["rule_source"] = rule_results.apply(lambda x: x[1] if x else None)
    
    has_rule = df["rule_label"].notna()
    print(f"Rule coverage: {has_rule.sum():,} ({has_rule.mean()*100:.1f}%)")
    
    print("\nCombining labels...")
    
    final_topics = []
    final_confs = []
    label_sources = []
    
    n_agree = 0
    n_llm_only = 0
    n_rule_only = 0
    n_disagree = 0
    n_unlabeled = 0
    
    for _, row in df.iterrows():
        llm_label = row.get("llm_label", None)
        llm_conf = row.get("llm_confidence", 0.0)
        rule_label = row.get("rule_label", None)
        
        has_l = pd.notna(llm_label) and llm_label in TOPICS
        has_r = pd.notna(rule_label) and rule_label in TOPICS
        
        if has_l and has_r:
            if llm_label == rule_label:
                final_topics.append(llm_label)
                final_confs.append(max(float(llm_conf), 0.9))
                label_sources.append("llm+rule_agree")
                n_agree += 1
            else:
                final_topics.append(llm_label)
                final_confs.append(float(llm_conf) * 0.8)
                label_sources.append("llm+rule_disagree")
                n_disagree += 1
        elif has_l:
            final_topics.append(llm_label)
            final_confs.append(float(llm_conf))
            label_sources.append("llm_only")
            n_llm_only += 1
        elif has_r:
            final_topics.append(rule_label)
            final_confs.append(0.6)
            label_sources.append("rule_only")
            n_rule_only += 1
        else:
            final_topics.append(None)
            final_confs.append(0.0)
            label_sources.append("none")
            n_unlabeled += 1
    
    df["topic_label"] = final_topics
    df["topic_confidence"] = final_confs
    df["label_source"] = label_sources
    
    total_labeled = n_agree + n_llm_only + n_rule_only + n_disagree
    print(f"\n=== LABEL COMBINATION STATS ===")
    print(f"  LLM + Rules agree:    {n_agree:>8,} ({n_agree/max(total_labeled,1)*100:.1f}%)")
    print(f"  LLM only:             {n_llm_only:>8,} ({n_llm_only/max(total_labeled,1)*100:.1f}%)")
    print(f"  Rules only:           {n_rule_only:>8,} ({n_rule_only/max(total_labeled,1)*100:.1f}%)")
    print(f"  LLM + Rules disagree: {n_disagree:>8,} ({n_disagree/max(total_labeled,1)*100:.1f}%)")
    print(f"  Unlabeled:            {n_unlabeled:>8,}")
    
    both_labeled = df[df["llm_label"].notna() & df["rule_label"].notna()]
    if len(both_labeled) > 0:
        from sklearn.metrics import cohen_kappa_score
        kappa = cohen_kappa_score(
            both_labeled["llm_label"].astype(str),
            both_labeled["rule_label"].astype(str)
        )
        agreement_pct = (both_labeled["llm_label"] == both_labeled["rule_label"]).mean() * 100
        print(f"\n=== INTER-ANNOTATOR AGREEMENT ===")
        print(f"  Items with both labels: {len(both_labeled):,}")
        print(f"  Raw agreement: {agreement_pct:.1f}%")
        print(f"  Cohen's Kappa: {kappa:.4f}")
    
    df_labeled = df[df["topic_label"].notna()].copy()
    df_filtered = df_labeled[df_labeled["topic_confidence"] >= min_confidence].copy()
    
    print(f"\n=== CONFIDENCE FILTERING (>= {min_confidence}) ===")
    print(f"  Before: {len(df_labeled):,}")
    print(f"  After:  {len(df_filtered):,} ({len(df_filtered)/max(len(df_labeled),1)*100:.1f}%)")
    
    print(f"\n=== FINAL LABEL DISTRIBUTION ===")
    print(df_filtered["topic_label"].value_counts())
    
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    cols_to_keep = [c for c in df_filtered.columns if not c.startswith("_") and c not in ["llm_label", "llm_confidence", "llm_reason", "rule_label", "rule_source", "label_source"]]
    df_filtered = df_filtered[cols_to_keep]
    
    return df_filtered

def prepare_splits(df: pd.DataFrame, val_ratio: float = 0.1, test_ratio: float = 0.1, seed: int = 42):
    from sklearn.model_selection import train_test_split
    
    train_val, test = train_test_split(
        df, test_size=test_ratio, stratify=df["topic_label"], random_state=seed
    )
    
    val_rel_ratio = val_ratio / (1.0 - test_ratio)
    train, val = train_test_split(
        train_val, test_size=val_rel_ratio, stratify=train_val["topic_label"], random_state=seed
    )
    
    train.to_parquet(OUTPUT_PATH / "train.parquet", index=False)
    val.to_parquet(OUTPUT_PATH / "val.parquet", index=False)
    test.to_parquet(OUTPUT_PATH / "test.parquet", index=False)
    
    print(f"\nSplits created:")
    print(f"Train: {len(train):,} ({len(train)/len(df)*100:.1f}%)")
    print(f"Val:   {len(val):,} ({len(val)/len(df)*100:.1f}%)")
    print(f"Test:  {len(test):,} ({len(test)/len(df)*100:.1f}%)")
    
    return train, val, test

def create_esg_subset(df: pd.DataFrame):
    print(f"\n=== CREATING ESG SUBSET ===")
    esg_df = df[df["topic_label"] != "Non_ESG"].copy()

    esg_path = Path("data/corpus/esg_sentences.parquet")
    esg_path.parent.mkdir(parents=True, exist_ok=True)
    esg_df.to_parquet(esg_path, index=False)
    
    print(f"Total ESG sentences: {len(esg_df):,}")
    print(esg_df["topic_label"].value_counts())
    print(f"Saved: {esg_path}")

if __name__ == "__main__":
    df = create_hybrid_labels()
    if df is not None:
        create_esg_subset(df)
        prepare_splits(df)
