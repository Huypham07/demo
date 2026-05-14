"""
ESG Evidence Detection Module
==============================

Pattern-based detection of evidence elements in ESG disclosure text.

Evidence Quality Hierarchy (GRI Standards, 2021; Cho et al., 2012):
----------------------------------------------------------------------
Tier 1 — Third_party (w=0.35): External verification/certification.
    Strongest substantiation — independent, verifiable. (GRI 2-5)
Tier 2 — KPI (w=0.30): Quantitative metrics with ESG relevance.
    Measurable, comparable, falsifiable. (GRI 302/305)
Tier 3 — Standard (w=0.20): References to normative frameworks.
    Demonstrates awareness and alignment. (GRI 2-23)
Tier 4 — Time_bound (w=0.15): Temporal specificity.
    Adds concreteness but not independently verifiable. (GRI 3-3)

References:
    GRI (2021). GRI Universal Standards 2021.
    Cho, C.H., et al. (2012). "The Role of Environmental Disclosure
        Quality." J. Account. Public Policy, 31(1), 73-90.
    Patten, D.M. (2002). "The relation between environmental performance
        and environmental disclosure." Accounting, Orgs & Society.
"""

import re
import pandas as pd
from pathlib import Path
from dataclasses import dataclass

@dataclass
class EvidencePattern:
    name: str
    patterns: list[str]
    weight: float

EVIDENCE_TYPES = {
    "Third_party": EvidencePattern(
        name="Third_party",
        weight=0.35,
        patterns=[
            r"\b(kiểm toán|audited?|xác nhận|verified|certified)\b",
            r"\b(chứng nhận|certification|accreditation)\b",
            r"\b(đánh giá độc lập|independent assessment)\b",
            r"\b(Deloitte|PwC|KPMG|EY|Ernst\s*&\s*Young)\b",
            r"\b(Bureau Veritas|DNV|SGS)\b",
            r"\b(FiinRatings|Vietnam Credit)\b",
            r"\b(giải thưởng|award|recognition)\b",
            r"\b(top\s*\d+|ranking|xếp hạng)\b",
        ],
    ),

    "KPI": EvidencePattern(
        name="KPI",
        weight=0.30,
        patterns=[
            r"\d+[\.,]?\d*\s*(%|phần trăm|percent)",
            r"\d+[\.,]?\d*\s*(tỷ|triệu)\s*(đồng|VND|USD)?",
            r"\d+[\.,]?\d*\s*(tấn|kg|ton|tonnes?)\s*(CO2|carbon)?",
            r"\d+[\.,]?\d*\s*(kWh|MWh|GWh|MW)",
            r"\d+[\.,]?\d*\s*(m2|m3|ha|hecta)",
            r"\d+[\.,]?\d*\s*(người|nhân viên|CBNV|khách hàng)",
            r"(giảm|tăng|đạt|tiết kiệm|cắt giảm)\s+\d+[\.,]?\d*\s*%",
            r"(so với|compared to).*\d+",
            r"Scope\s*[123].*\d+",
        ],
    ),
    
    "Standard": EvidencePattern(
        name="Standard",
        weight=0.20,
        patterns=[
            r"\b(GRI\s*\d*|Global Reporting Initiative)\b",
            r"\b(SBTi|Science Based Targets?)\b",
            r"\b(ISO\s*\d+|ISO\s*14001|ISO\s*26000)\b",
            r"\b(SDG|Sustainable Development Goals?)\b",
            r"\b(Net[\-\s]?Zero|carbon neutral|trung hòa carbon)\b",
            r"\b(Paris Agreement|COP\d+)\b",
            r"\b(CDP|Carbon Disclosure Project)\b",
            r"\b(TCFD|Task Force on Climate)\b",
            r"\b(UN Global Compact|UNGC)\b",
            r"\b(ESG rating|xếp hạng ESG)\b",
            r"\b(Thông tư\s*\d+|Nghị định\s*\d+)\b",
            r"\b(NHNN|Ngân hàng Nhà nước)\b",
        ],
    ),
    
    "Time_bound": EvidencePattern(
        name="Time_bound",
        weight=0.15,
        patterns=[
            r"\b(năm|year)\s*(20\d{2})\b",
            r"\b(trong năm|in)\s*(20\d{2})\b",
            r"\b(đến|by|until)\s*(20\d{2})\b",
            r"\b(giai đoạn|period)\s*(20\d{2})\s*[-–]\s*(20\d{2})\b",
            r"\b(Q[1-4]|quý\s*[1-4IViv]+)[\/\s]*(20\d{2})\b",
            r"\b(tháng\s*\d+|month\s*\d+)[\/\s]*(20\d{2})\b",
            r"\b(trong\s+\d+\s*(năm|tháng|quý))\b",
            r"\b(mục tiêu|target)\s*(20\d{2})\b",
        ],
    ),
    
}

QUALITY_WEIGHTS = {t: p.weight for t, p in EVIDENCE_TYPES.items()}

VALID_EVIDENCE_TYPES = list(EVIDENCE_TYPES.keys())

def detect_evidence(text: str, context: str = "") -> dict:
    """
    Detect evidence elements in text using pattern matching.

    Returns:
        dict with:
        - has_evidence: bool
        - evidence_types: list of detected types
        - evidence_matches: dict of type -> list of matches
    """
    full_text = f"{context} {text}".lower() if context else text.lower()
    
    evidence_types = []
    evidence_matches = {}
    
    for etype, epattern in EVIDENCE_TYPES.items():
        matches = []
        for pattern in epattern.patterns:
            found = re.findall(pattern, text, re.IGNORECASE)
            if found:
                matches.extend(found if isinstance(found[0], str) else [m[0] if isinstance(m, tuple) else m for m in found])
        
        if matches:
            evidence_types.append(etype)
            evidence_matches[etype] = list(set(matches))[:5]
    
    return {
        "has_evidence": len(evidence_types) > 0,
        "evidence_types": evidence_types,
        "evidence_matches": evidence_matches,
    }

def calculate_quality_score(evidence_types: list) -> float:
    """
    Calculate evidence quality from detected types (GRI hierarchy).

    Quality = Σ(weights of found types) / Σ(all weights)

    This is only the regex-based quality signal. The full Evidence
    Strength (ES) is computed in the EWRI module by combining quality,
    calibrated similarity, and directional NLI.
    """
    if not evidence_types:
        return 0.0

    total = sum(QUALITY_WEIGHTS.get(et, 0.0) for et in evidence_types)
    max_possible = sum(QUALITY_WEIGHTS.values())
    return min(total / max_possible, 1.0) if max_possible > 0 else 0.0

def extract_kpi_values(text: str) -> list[str]:
    """Extract specific KPI values from text."""
    kpi_patterns = [
        r"(\d+[\.,]?\d*\s*%)",
        r"(\d+[\.,]?\d*\s*(tỷ|triệu)\s*đồng)",
        r"(\d+[\.,]?\d*\s*(tấn|kg)\s*CO2?)",
        r"(\d+[\.,]?\d*\s*(kWh|MWh))",
    ]
    
    values = []
    for pattern in kpi_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for m in matches:
            val = m[0] if isinstance(m, tuple) else m
            values.append(val.strip())
    
    return list(set(values))[:10]

def process_dataframe(df: pd.DataFrame, text_col: str = "sentence") -> pd.DataFrame:
    """Process entire dataframe and add evidence detection columns."""
    print(f"Detecting evidence in {len(df):,} sentences...")

    results = []
    for _, row in df.iterrows():
        text = str(row[text_col])
        ctx = f"{row.get('ctx_prev', '')} {row.get('ctx_next', '')}"

        result = detect_evidence(text, ctx)
        result["kpi_values"] = extract_kpi_values(text)
        results.append(result)

    df = df.copy()
    df["has_evidence"] = [r["has_evidence"] for r in results]
    df["evidence_types"] = [r["evidence_types"] for r in results]
    df["kpi_values"] = [r["kpi_values"] for r in results]

    return df