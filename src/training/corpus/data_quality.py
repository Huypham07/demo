import re
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
from typing import Optional

def check_length(
    df: pd.DataFrame,
    text_col: str = "sentence",
    min_length: int = 20,
    max_length: int = 500,
) -> pd.DataFrame:
    df = df.copy()
    lengths = df[text_col].str.len()
    df["_char_length"] = lengths
    df["_length_ok"] = (lengths >= min_length) & (lengths <= max_length)
    
    too_short = (lengths < min_length).sum()
    too_long = (lengths > max_length).sum()
    
    print(f"[Length] Too short (<{min_length}): {too_short} ({100*too_short/len(df):.1f}%)")
    print(f"[Length] Too long (>{max_length}): {too_long} ({100*too_long/len(df):.1f}%)")
    print(f"[Length] OK: {df['_length_ok'].sum()} ({100*df['_length_ok'].sum()/len(df):.1f}%)")
    
    return df

def _char_ngrams(text: str, n: int = 3) -> set:
    """Extract character n-grams from text."""
    text = text.lower().strip()
    return set(text[i:i+n] for i in range(len(text) - n + 1))

def _jaccard_similarity(set_a: set, set_b: set) -> float:
    """Compute Jaccard similarity between two sets."""
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0

def check_duplicates(
    df: pd.DataFrame,
    text_col: str = "sentence",
    threshold: float = 0.9,
    ngram_size: int = 3,
    sample_size: Optional[int] = None,
) -> pd.DataFrame:
    df = df.copy()
    texts = df[text_col].tolist()
    n = len(texts)
    
    if sample_size and n > sample_size:
        print(f"[Duplicates] Sampling {sample_size}/{n} for duplicate check")
        sample_idx = np.random.choice(n, sample_size, replace=False)
    else:
        sample_idx = range(n)
    
    ngram_sets = {}
    for i in sample_idx:
        ngram_sets[i] = _char_ngrams(texts[i], ngram_size)
    
    duplicate_mask = set()
    seen = []
    
    for i in sample_idx:
        is_dup = False
        for j in seen[-50:]:
            sim = _jaccard_similarity(ngram_sets[i], ngram_sets[j])
            if sim >= threshold:
                duplicate_mask.add(i)
                is_dup = True
                break
        if not is_dup:
            seen.append(i)
    
    df["_is_duplicate"] = False
    for idx in duplicate_mask:
        df.iloc[idx, df.columns.get_loc("_is_duplicate")] = True
    
    dup_count = len(duplicate_mask)
    print(f"[Duplicates] Near-duplicates found: {dup_count} ({100*dup_count/len(df):.1f}%)")
    
    return df

OCR_ARTIFACT_PATTERNS = [
    re.compile(r'\b[bcdfghjklmnpqrstvwxyz]{4,}\b', re.IGNORECASE),
    re.compile(r'[�\ufffd\x00-\x08\x0b\x0c\x0e-\x1f]'),
    re.compile(r'[^\w\s\.,;:!?\-\(\)\"\'%/]{3,}'),
    re.compile(r'(.)\1{4,}'),
    re.compile(r'[a-zA-Z][àáảãạăắằẳẵặâấầẩẫậ][a-zA-Z]{2}[àáảãạ]', re.IGNORECASE),
]

def check_ocr_quality(
    df: pd.DataFrame,
    text_col: str = "sentence",
    max_artifact_ratio: float = 0.1,
) -> pd.DataFrame:
    df = df.copy()
    
    artifact_counts = []
    for text in df[text_col]:
        text = str(text)
        count = 0
        for pattern in OCR_ARTIFACT_PATTERNS:
            count += len(pattern.findall(text))
        ratio = count / max(len(text), 1)
        artifact_counts.append(ratio)
    
    df["_ocr_artifact_ratio"] = artifact_counts
    df["_ocr_clean"] = df["_ocr_artifact_ratio"] <= max_artifact_ratio
    
    dirty = (~df["_ocr_clean"]).sum()
    print(f"[OCR] Sentences with artifacts: {dirty} ({100*dirty/len(df):.1f}%)")
    
    return df

def check_class_balance(
    df: pd.DataFrame,
    label_col: str = "final_topic",
    min_class_pct: float = 5.0,
) -> dict:
    if label_col not in df.columns:
        print(f"[Balance] Column '{label_col}' not found. Skipping.")
        return {}
    
    counts = df[label_col].value_counts()
    total = len(df)
    
    print(f"\n[Balance] Class distribution for '{label_col}':")
    imbalanced_classes = []
    
    for label, count in counts.items():
        pct = 100 * count / total
        flag = " ⚠️ IMBALANCED" if pct < min_class_pct else ""
        print(f"  {label:20s}: {count:6,} ({pct:5.1f}%){flag}")
        if pct < min_class_pct:
            imbalanced_classes.append(label)
    
    balance_stats = {
        "total": total,
        "n_classes": len(counts),
        "distribution": counts.to_dict(),
        "imbalanced_classes": imbalanced_classes,
        "is_balanced": len(imbalanced_classes) == 0,
    }
    
    if imbalanced_classes:
        print(f"  ⚠️ {len(imbalanced_classes)} classes below {min_class_pct}% threshold: "
              f"{imbalanced_classes}")
    else:
        print(f"  ✅ All classes above {min_class_pct}% threshold")
    
    return balance_stats

def run_quality_checks(
    df: pd.DataFrame,
    text_col: str = "sentence",
    label_col: str = "final_topic",
    min_length: int = 20,
    max_length: int = 500,
    dup_threshold: float = 0.9,
    ocr_max_ratio: float = 0.1,
    min_class_pct: float = 5.0,
) -> tuple[pd.DataFrame, dict]:
    """
    Run all quality checks and produce a summary report.
    
    Returns:
        (df_with_flags, quality_report)
    """
    print("=" * 60)
    print("DATA QUALITY VALIDATION REPORT")
    print("=" * 60)
    print(f"Total sentences: {len(df):,}")
    print()
    
    df = check_length(df, text_col, min_length, max_length)
    
    df = check_duplicates(df, text_col, dup_threshold, sample_size=min(len(df), 10000))
    
    df = check_ocr_quality(df, text_col, ocr_max_ratio)
    
    balance = check_class_balance(df, label_col, min_class_pct)
    
    clean_mask = df["_length_ok"] & ~df["_is_duplicate"] & df["_ocr_clean"]
    clean_count = clean_mask.sum()
    
    print(f"\n{'='*60}")
    print(f"OVERALL QUALITY")
    print(f"{'='*60}")
    print(f"Clean sentences: {clean_count:,} / {len(df):,} ({100*clean_count/len(df):.1f}%)")
    print(f"Flagged for removal: {len(df) - clean_count:,}")
    
    report = {
        "total": len(df),
        "clean": int(clean_count),
        "clean_pct": round(100 * clean_count / len(df), 1),
        "length_failed": int((~df["_length_ok"]).sum()),
        "duplicates": int(df["_is_duplicate"].sum()),
        "ocr_artifacts": int((~df["_ocr_clean"]).sum()),
        "class_balance": balance,
    }
    
    return df, report

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/corpus/sentences.parquet")
    parser.add_argument("--label-col", type=str, default="final_topic")
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"File not found: {input_path}")
    else:
        df = pd.read_parquet(input_path)
        df_checked, report = run_quality_checks(df, label_col=args.label_col)
        
        output_path = input_path.parent / f"{input_path.stem}_quality_checked.parquet"
        df_checked.to_parquet(output_path, index=False)
        print(f"\nSaved: {output_path}")
