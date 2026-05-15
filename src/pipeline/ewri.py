import math
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional

VALID_EVIDENCE_TYPES = ["KPI", "Standard", "Time_bound", "Third_party"]

ACTION_PENALTY = {
    "Implemented": 0.02,
    "Planning": 0.15,
    "Indeterminate": 0.55,
}

EVIDENCE_SENSITIVITY = {
    "Implemented": 1.00,
    "Planning": 0.85,
    "Indeterminate": 0.50,
}

CONTRADICTION_AMPLIFIER = 1.8

ESG_TOPICS = ["E", "S_labor", "S_community", "S_product", "G"]

def configure_from_dict(cfg: Optional[dict]) -> None:
    configure_ewri(cfg)

def configure_ewri(cfg: Optional[dict]) -> None:
    if not cfg:
        return
    global ACTION_PENALTY, EVIDENCE_SENSITIVITY, CONTRADICTION_AMPLIFIER
    if "action_penalty" in cfg:
        ACTION_PENALTY = {**ACTION_PENALTY, **cfg["action_penalty"]}
    if "evidence_sensitivity" in cfg:
        EVIDENCE_SENSITIVITY = {**EVIDENCE_SENSITIVITY, **cfg["evidence_sensitivity"]}
    if "contradiction_amplifier" in cfg:
        CONTRADICTION_AMPLIFIER = float(cfg["contradiction_amplifier"])

@dataclass
class EWRIScore:
    bank: str
    year: int
    total_sentences: int

    implemented: int
    planning: int
    indeterminate: int

    with_evidence: int
    without_evidence: int

    implemented_ratio: float
    planning_ratio: float
    indeterminate_ratio: float
    evidence_ratio: float
    avg_evidence_strength: float

    ewri: float

    contribution_implemented: float
    contribution_planning: float
    contribution_indeterminate: float

    topic_breakdown: dict = field(default_factory=dict)
    topic_entropy: float = 0.0
    topic_coverage_index: float = 0.0

    sentence_risks: List[dict] = field(default_factory=list)

def compute_evidence_score(row: pd.Series) -> float:
    has_evidence = bool(row.get("has_evidence", False))
    if not has_evidence:
        return 0.0
    nli_label = str(row.get("nli_label", "") or "")
    # support = 0 only when no evidence OR contradiction; neutral counts as support
    return 0.0 if nli_label == "contradiction" else 1.0

def compute_washing_risk(
    action_label: str, evidence_strength: float, nli_label: str = ""
) -> float:
    p = ACTION_PENALTY.get(action_label, 0.55)
    lam = EVIDENCE_SENSITIVITY.get(action_label, 0.50)
    es = max(0.0, min(1.0, evidence_strength))
    wrs = p * (1.0 - lam * es)

    if nli_label == "contradiction":
        wrs = min(1.0, wrs * CONTRADICTION_AMPLIFIER)

    return max(0.0, min(1.0, wrs))

def calculate_topic_entropy(topic_counts: dict) -> float:
    total = sum(topic_counts.values())
    if total == 0:
        return 0.0
    probs = [c / total for c in topic_counts.values() if c > 0]
    if len(probs) <= 1:
        return 0.0
    entropy = -sum(p * math.log2(p) for p in probs)
    max_entropy = math.log2(len(probs))
    return entropy / max_entropy if max_entropy > 0 else 0.0

def enrich_with_risk_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["es_combined"] = df.apply(compute_evidence_score, axis=1)

    df["wrs"] = df.apply(
        lambda row: compute_washing_risk(
            row["action_label"], row["es_combined"],
            str(row.get("nli_label", "") or "")
        ),
        axis=1,
    )

    return df

def _topic_breakdown(group: pd.DataFrame, topic_col: str) -> dict:
    breakdown = {}
    for topic in ESG_TOPICS:
        tdf = group[group[topic_col] == topic]
        n = len(tdf)
        if n < 3:
            breakdown[topic] = {"ewri": None, "n": n}
            continue
        t_ewri = tdf["wrs"].mean() * 100
        impl = int((tdf["action_label"] == "Implemented").sum())
        indet = int((tdf["action_label"] == "Indeterminate").sum())
        has_ev = int(tdf["has_evidence"].sum()) if "has_evidence" in tdf.columns else 0
        breakdown[topic] = {
            "ewri": round(t_ewri, 2),
            "n": n,
            "implemented_pct": round(impl / n * 100, 1),
            "indeterminate_pct": round(indet / n * 100, 1),
            "evidence_rate": round(has_ev / n * 100, 1),
        }
    return breakdown

def calculate_bank_year_ewri(df: pd.DataFrame) -> list[EWRIScore]:
    topic_col = "topic_label" if "topic_label" in df.columns else None

    if "wrs" not in df.columns:
        df = enrich_with_risk_scores(df)

    scores = []

    for (bank, year), group in df.groupby(["bank", "year"]):
        N = len(group)
        if N == 0:
            continue

        impl = int((group["action_label"] == "Implemented").sum())
        plan = int((group["action_label"] == "Planning").sum())
        indet = int((group["action_label"] == "Indeterminate").sum())

        with_ev = int((group["es_combined"] > 0).sum()) if "es_combined" in group.columns else 0

        ewri_new = group["wrs"].mean() * 100

        impl_mask = group["action_label"] == "Implemented"
        plan_mask = group["action_label"] == "Planning"
        indet_mask = group["action_label"] == "Indeterminate"
        c_impl = group.loc[impl_mask, "wrs"].sum() / N * 100 if impl > 0 else 0.0
        c_plan = group.loc[plan_mask, "wrs"].sum() / N * 100 if plan > 0 else 0.0
        c_indet = group.loc[indet_mask, "wrs"].sum() / N * 100 if indet > 0 else 0.0

        avg_es = group["es_combined"].mean()

        topic_bd = {}
        topic_counts = {}
        if topic_col:
            topic_bd = _topic_breakdown(group, topic_col)
            for t in ESG_TOPICS:
                topic_counts[t] = int((group[topic_col] == t).sum())

        te = calculate_topic_entropy(topic_counts) if topic_counts else 0.0

        sent_risks = []
        for _, row in group.iterrows():
            sent_risks.append({
                "sent_id": row.get("sent_id", ""),
                "sentence": str(row.get("sentence", "")),
                "block_text": str(row.get("block_text", "")),
                "block_prev_text": str(row.get("block_prev_text", "")),
                "block_next_text": str(row.get("block_next_text", "")),
                "section_title": str(row.get("section_title", "")),
                "block_type": str(row.get("block_type", "")),
                "action_label": row.get("action_label", "Unknown"),
                "action_confidence": round(float(row.get("action_confidence", 0.0)), 3),
                "has_evidence": bool(row.get("has_evidence", False)),
                "evidence_strength": round(float(row.get("es_combined", 0.0)), 3),
                "washing_risk": round(float(row.get("wrs", 0.0)), 3),
                "topic": row.get(topic_col, "Unknown") if topic_col else "Unknown",
                "evidence_types": list(row.get("evidence_types", [])) if isinstance(row.get("evidence_types", []), (list, np.ndarray)) else [],
                "nli_label": row.get("nli_label", ""),
                "best_evidence": str(row.get("best_evidence", "")),
            })
        sent_risks.sort(key=lambda x: x["washing_risk"], reverse=True)

        scores.append(EWRIScore(
            bank=bank, year=year, total_sentences=N,
            implemented=impl, planning=plan, indeterminate=indet,
            with_evidence=with_ev, without_evidence=N - with_ev,
            implemented_ratio=round(impl / N, 3),
            planning_ratio=round(plan / N, 3),
            indeterminate_ratio=round(indet / N, 3),
            evidence_ratio=round(with_ev / N, 3),
            avg_evidence_strength=round(avg_es, 3),
            ewri=round(ewri_new, 2),
            contribution_implemented=round(c_impl, 2),
            contribution_planning=round(c_plan, 2),
            contribution_indeterminate=round(c_indet, 2),
            topic_breakdown=topic_bd,
            topic_entropy=round(te, 3),
            topic_coverage_index=round(1.0 - te, 3),
            sentence_risks=sent_risks,
        ))

    return scores

def scores_to_dataframe(scores: list[EWRIScore]) -> pd.DataFrame:
    data = []
    for s in scores:
        data.append({
            "bank": s.bank, "year": s.year,
            "total_sentences": s.total_sentences,
            "implemented": s.implemented, "planning": s.planning,
            "indeterminate": s.indeterminate,
            "with_evidence": s.with_evidence, "without_evidence": s.without_evidence,
            "implemented_ratio": s.implemented_ratio,
            "planning_ratio": s.planning_ratio,
            "indeterminate_ratio": s.indeterminate_ratio,
            "evidence_ratio": s.evidence_ratio,
            "avg_evidence_strength": s.avg_evidence_strength,
            "ewri": s.ewri,
            "contrib_implemented": s.contribution_implemented,
            "contrib_planning": s.contribution_planning,
            "contrib_indeterminate": s.contribution_indeterminate,
            "topic_entropy": s.topic_entropy,
            "topic_coverage_index": s.topic_coverage_index,
        })
    return pd.DataFrame(data)

def print_ewri_summary(df_scores: pd.DataFrame):
    print("\n" + "=" * 70)
    print("EWRI SUMMARY (Interaction-Based Formula)")
    print("=" * 70)

    ewri = df_scores["ewri"]
    print(f"\nTotal bank-years: {len(df_scores)}")
    print(f"EWRI Range: [{ewri.min():.2f}, {ewri.max():.2f}]")
    print(f"EWRI Mean - Std: {ewri.mean():.2f} - {ewri.std():.2f}")
    print(f"Quartiles:  Q1={ewri.quantile(0.25):.2f}  "
          f"Median={ewri.median():.2f}  Q3={ewri.quantile(0.75):.2f}")

    ewri_mean = max(ewri.mean(), 1e-9)
    print(f"\nEWRI Decomposition (average contribution):")
    print(f"  Indeterminate: {df_scores['contrib_indeterminate'].mean():.2f} pts "
          f"({df_scores['contrib_indeterminate'].mean() / ewri_mean * 100:.1f}%)")
    print(f"  Planning:      {df_scores['contrib_planning'].mean():.2f} pts "
          f"({df_scores['contrib_planning'].mean() / ewri_mean * 100:.1f}%)")
    print(f"  Implemented:   {df_scores['contrib_implemented'].mean():.2f} pts "
          f"({df_scores['contrib_implemented'].mean() / ewri_mean * 100:.1f}%)")

    sorted_s = df_scores.sort_values("ewri", ascending=False)
    print(f"\nTop 5 Highest EWRI:")
    for _, r in sorted_s.head(5).iterrows():
        print(f"  {r['bank']:15s} {r['year']}  EWRI={r['ewri']:5.1f}")

    print(f"\nTop 5 Lowest EWRI:")
    for _, r in sorted_s.tail(5).iterrows():
        print(f"  {r['bank']:15s} {r['year']}  EWRI={r['ewri']:5.1f}")

if __name__ == "__main__":
    import sys
    input_path = sys.argv[1] if len(sys.argv) > 1 else "data/corpus/actionability_sentences.parquet"
    df = pd.read_parquet(input_path)
    df = enrich_with_risk_scores(df)
    scores = calculate_bank_year_ewri(df)
    df_scores = scores_to_dataframe(scores).sort_values("ewri", ascending=False)
    print_ewri_summary(df_scores)
