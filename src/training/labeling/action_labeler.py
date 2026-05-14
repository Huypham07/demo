"""
Actionability Label Aggregation - LLM Primary + Rule Validation.

Same strategy as topic hybrid_labeler:
    1. LLM labels ALL ESG sentences (primary source)
    2. Rules (Bloom/Hyland) validate LLM labels
    3. Agreement-based confidence calibration
"""

import numpy as np
import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.labeling.grounded_rules import match_actionability_grounded

ESG_SENTENCES_PATH = Path("data/corpus/esg_sentences.parquet")
LLM_LABELS_PATH = Path("data/labels/action/llm_prelabels.parquet")
OUTPUT_PATH = Path("data/labels/action")

LABELS = ["Implemented", "Planning", "Indeterminate"]

def create_hybrid_labels(min_confidence: float = 0.5) -> pd.DataFrame:
    """
    Create actionability labels: LLM primary + rules validation.
    """
    
    print("Loading ESG sentences...")
    if not ESG_SENTENCES_PATH.exists():
        print("ERROR: Run topic classification pipeline first.")
        return None
    
    df = pd.read_parquet(ESG_SENTENCES_PATH)
    print(f"ESG sentences: {len(df):,}")
    
    print("Loading LLM labels...")
    if not LLM_LABELS_PATH.exists():
        print(f"ERROR: LLM labels not found at {LLM_LABELS_PATH}")
        print("Run: python -m src.actionability.llm_labeler")
        return None
    
    llm = pd.read_parquet(LLM_LABELS_PATH)
    print(f"LLM labels: {len(llm):,}")
    
    df = df.merge(
        llm[["sent_id", "llm_label", "llm_confidence", "llm_reason"]],
        on="sent_id", how="left"
    )
    
    has_llm = df["llm_label"].notna()
    print(f"LLM coverage: {has_llm.sum():,} ({has_llm.mean()*100:.1f}%)")
    
    print("\nApplying Bloom/Hyland rules...")
    rule_results = df.apply(
        lambda row: match_actionability_grounded(str(row.get("sentence", ""))),
        axis=1
    )
    
    df["rule_label"] = rule_results.apply(lambda x: x[0] if x else None)
    df["rule_source"] = rule_results.apply(lambda x: x[1] if x else None)
    
    has_rule = df["rule_label"].notna()
    print(f"Rule coverage: {has_rule.sum():,} ({has_rule.mean()*100:.1f}%)")
    
    print("\nCombining labels...")
    
    final_actions = []
    final_confs = []
    label_sources = []
    counters = {"agree": 0, "llm_only": 0, "rule_only": 0, "disagree": 0, "none": 0}
    
    for _, row in df.iterrows():
        llm_label = row.get("llm_label", None)
        llm_conf = row.get("llm_confidence", 0.0)
        rule_label = row.get("rule_label", None)
        
        has_l = pd.notna(llm_label) and llm_label in LABELS
        has_r = pd.notna(rule_label) and rule_label in LABELS
        
        if has_l and has_r:
            if llm_label == rule_label:
                final_actions.append(llm_label)
                final_confs.append(max(float(llm_conf), 0.9))
                label_sources.append("llm+rule_agree")
                counters["agree"] += 1
            else:
                final_actions.append(llm_label)
                final_confs.append(float(llm_conf) * 0.8)
                label_sources.append("llm+rule_disagree")
                counters["disagree"] += 1
        elif has_l:
            final_actions.append(llm_label)
            final_confs.append(float(llm_conf))
            label_sources.append("llm_only")
            counters["llm_only"] += 1
        elif has_r:
            final_actions.append(rule_label)
            final_confs.append(0.6)
            label_sources.append("rule_only")
            counters["rule_only"] += 1
        else:
            final_actions.append(None)
            final_confs.append(0.0)
            label_sources.append("none")
            counters["none"] += 1
    
    df["action_label"] = final_actions
    df["action_confidence"] = final_confs
    df["label_source"] = label_sources
    
    total = sum(v for k, v in counters.items() if k != "none")
    print(f"\n=== LABEL COMBINATION STATS ===")
    for key, val in counters.items():
        pct = val / max(total, 1) * 100 if key != "none" else 0
        print(f"  {key:20s}: {val:>8,} ({pct:.1f}%)")
    
    both = df[df["llm_label"].notna() & df["rule_label"].notna()]
    if len(both) > 0:
        from sklearn.metrics import cohen_kappa_score
        kappa = cohen_kappa_score(both["llm_label"].astype(str), both["rule_label"].astype(str))
        agree_pct = (both["llm_label"] == both["rule_label"]).mean() * 100
        print(f"\n=== INTER-ANNOTATOR AGREEMENT ===")
        print(f"  Items: {len(both):,}, Agreement: {agree_pct:.1f}%, Kappa: {kappa:.4f}")
    
    df_labeled = df[df["action_label"].notna()].copy()
    df_filtered = df_labeled[df_labeled["action_confidence"] >= min_confidence].copy()
    
    print(f"\n=== CONFIDENCE FILTERING (>= {min_confidence}) ===")
    print(f"  Before: {len(df_labeled):,}, After: {len(df_filtered):,}")
    
    print(f"\n=== FINAL DISTRIBUTION ===")
    print(df_filtered["action_label"].value_counts())
    
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    cols_to_keep = [c for c in df_filtered.columns if not c.startswith("_") and c not in ["llm_label", "llm_confidence", "llm_reason", "rule_label", "rule_source", "label_source"]]
    df_filtered = df_filtered[cols_to_keep]
    
    return df_filtered

def prepare_splits(df: pd.DataFrame, val_ratio: float = 0.1, test_ratio: float = 0.1, seed: int = 42):
    """Split into train/val/test, stratified by action."""
    from sklearn.model_selection import train_test_split
    
    train_val, test = train_test_split(
        df, test_size=test_ratio, stratify=df["action_label"], random_state=seed
    )
    
    val_rel_ratio = val_ratio / (1.0 - test_ratio)
    train, val = train_test_split(
        train_val, test_size=val_rel_ratio, stratify=train_val["action_label"], random_state=seed
    )
    
    train.to_parquet(OUTPUT_PATH / "train.parquet", index=False)
    val.to_parquet(OUTPUT_PATH / "val.parquet", index=False)
    test.to_parquet(OUTPUT_PATH / "test.parquet", index=False)
    
    print(f"\nSplits created:")
    print(f"Train: {len(train):,} ({len(train)/len(df)*100:.1f}%)")
    print(f"Val:   {len(val):,} ({len(val)/len(df)*100:.1f}%)")
    print(f"Test:  {len(test):,} ({len(test)/len(df)*100:.1f}%)")
    
    return train, val, test

def create_action_subset(df: pd.DataFrame):
    print(f"\n=== CREATING ACTIONABILITY SUBSET ===")
    
    action_path = Path("data/corpus/actionability_sentences.parquet")
    action_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(action_path, index=False)
    
    print(f"Total actionability sentences: {len(df):,}")
    print(f"Saved: {action_path}")

if __name__ == "__main__":
    df = create_hybrid_labels()
    if df is not None:
        create_action_subset(df)
        prepare_splits(df)
