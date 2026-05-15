import re
import pandas as pd
from dataclasses import dataclass


@dataclass
class EvidencePattern:
    name: str
    patterns: list[str]


EVIDENCE_TYPES = {
    "Third_party": EvidencePattern(
        name="Third_party",
        patterns=[
            # Audit/assurance with explicit agent (highest precision)
            r"\b(kiểm toán|kiểm định|thẩm định).{0,20}(bởi|bên|đơn vị|tổ chức|công ty)\b",
            r"\b(xác nhận|xác thực|đảm bảo).{0,15}(bởi|từ|bên thứ ba|độc lập)\b",
            r"\b(audited?|verified|certified|assured?)\b",
            r"\b(đánh giá độc lập|independent (assessment|review|audit|assurance))\b",
            r"\b(bên thứ ba|third.?party|đơn vị độc lập)\b",
            # Named international assurance/audit firms
            r"\b(Deloitte|PwC|KPMG|EY|Ernst\s*&\s*Young|Grant Thornton|BDO)\b",
            r"\b(Bureau Veritas|DNV|DNV[\s-]GL|SGS|Intertek|TÜV|Lloyd.s Register)\b",
            # ESG rating and data providers
            r"\b(FiinRatings|FiinESG|Vietnam Credit|VIS Rating)\b",
            r"\b(Sustainalytics|MSCI ESG|S&P Global ESG|CDP Score)\b",
            # Certification and accreditation
            r"\b(chứng nhận|certification|accreditation)\b",
            # Awards with ESG context (reduces false positives from generic "award")
            r"\b(giải thưởng|award|recognition).{0,25}(ESG|bền vững|sustainability|môi trường|xanh)\b",
            r"\b(top\s*\d+|xếp hạng.{0,10}ESG|ESG (rating|score|index))\b",
        ],
    ),

    "KPI": EvidencePattern(
        name="KPI",
        patterns=[
            # Percentages — broadly reliable for ESG KPIs
            r"\d+[\.,]?\d*\s*(%|phần trăm|percent)",
            # Monetary values with currency (ESG investment/green lending)
            r"\d+[\.,]?\d*\s*(tỷ|triệu)\s*(đồng|VND|USD)\b",
            # Environmental units
            r"\d+[\.,]?\d*\s*(tấn|kg|ton|tonnes?)\s*(CO2|carbon|CO2e|eq)?\b",
            r"\d+[\.,]?\d*\s*(kWh|MWh|GWh|MW)\b",
            r"\d+[\.,]?\d*\s*m3\b.{0,15}(nước|water)",
            r"\d+[\.,]?\d*\s*(m2|ha|hecta|km2)\b",
            # Directional performance with percentage (most reliable ESG KPI signal)
            r"(giảm|tăng|đạt|tiết kiệm|cắt giảm)\s+\d+[\.,]?\d*\s*%",
            # Year-on-year comparison with measurable ESG units
            r"\d+[\.,]?\d*\s*(%|tỷ|triệu|tấn|kWh|MWh).{0,40}(so với|so sánh).{0,20}(năm|year)\s*(20\d{2})",
            # Social KPIs with explicit ESG action context (avoids bare employee count)
            r"\d[\d\.,]*\s*(người|nhân viên|CBNV)\b.{0,35}(được đào tạo|tham gia|đã tiếp cận|hưởng|được hỗ trợ|bồi dưỡng)",
            r"\d[\d\.,]*\s*(khách hàng|người dùng)\b.{0,35}(tiếp cận|được phục vụ|tài chính toàn diện|hỗ trợ vay)",
            # Scope emissions with measurable quantity
            r"Scope\s*[123].{0,40}\d+[\.,]?\d*\s*(tấn|tCO2|CO2e?)\b",
            # Banking-specific ESG KPIs
            r"(tỷ trọng|tỷ lệ|dư nợ).{0,20}(tín dụng xanh|xanh|bền vững|ESG).{0,20}\d+[\.,]?\d*\s*%",
            r"(số lượng|danh mục).{0,20}(sản phẩm xanh|khoản vay xanh|trái phiếu xanh).{0,20}\d",
        ],
    ),

    "Standard": EvidencePattern(
        name="Standard",
        patterns=[
            # International ESG reporting frameworks
            r"\b(GRI\s*\d+|GRI\s*Standards?|Global Reporting Initiative)\b",
            r"\b(SBTi|Science.?Based Targets?)\b",
            r"\b(ISO\s*14001|ISO\s*26000|ISO\s*45001|ISO\s*50001|ISO\s*\d{4,5})\b",
            r"\b(SASB|Sustainability Accounting Standards)\b",
            r"\b(IIRC|Integrated Reporting Framework)\b",
            r"\b(PRI|Principles for Responsible Investment)\b",
            r"\b(Equator Principles|Nguyên tắc Xích đạo)\b",
            # Climate and sustainability goals
            r"\b(SDG\s*\d+|Sustainable Development Goals?|Mục tiêu Phát triển Bền vững)\b",
            r"\b(Net[\-\s]?Zero|carbon neutral|trung hòa carbon|phát thải ròng bằng không)\b",
            r"\b(Paris Agreement|Hiệp định Paris|COP\s*\d+|UNFCCC)\b",
            r"\b(CDP|Carbon Disclosure Project)\b",
            r"\b(TCFD|Task Force on Climate.?related Financial)\b",
            r"\b(UN Global Compact|UNGC|Global Compact)\b",
            # Finance-sector-specific standards
            r"\b(Basel\s*[II]{1,3}|Basel\s*[23]|BCBS)\b",
            r"\b(IFRS\s*S[12]|International Sustainability Standards)\b",
            # Vietnamese regulations with ESG/green context
            r"\b(Thông tư\s*\d+[\/\-]\w*|Nghị định\s*\d+[\/\-]\w*)\b",
            r"\b(NHNN|Ngân hàng Nhà nước)\b.{0,50}(quy định|thông tư|chỉ thị|hướng dẫn).{0,30}(ESG|xanh|bền vững|môi trường|khí hậu)",
        ],
    ),

    "Time_bound": EvidencePattern(
        name="Time_bound",
        patterns=[
            # Specific year references
            r"\b(năm|year)\s*(20\d{2})\b",
            r"\btrong năm\s*(20\d{2})\b",
            r"\b(đến năm|đến|by|until)\s*(20\d{2})\b",
            # Periods and ranges
            r"\b(giai đoạn|period)\s*(20\d{2})\s*[-–]\s*(20\d{2})\b",
            # Quarter references
            r"\b(Q[1-4]|quý\s*[1-4IViv]+)[\/\s]*(20\d{2})\b",
            # Month references
            r"\btháng\s*\d{1,2}[\/\s]*(20\d{2})\b",
            # Relative time (duration)
            r"\b(trong\s+\d+\s*(năm|tháng|quý))\b",
            # Targets and milestones with year
            r"\b(mục tiêu|target).{0,15}(20\d{2})\b",
            # Roadmap/plan with year (forward-looking timelines)
            r"\b(lộ trình|roadmap|kế hoạch).{0,25}(20\d{2})\b",
        ],
    ),
}

VALID_EVIDENCE_TYPES = list(EVIDENCE_TYPES.keys())


def detect_evidence(text: str) -> dict:
    normalized = text.lower()

    evidence_types = []
    evidence_matches = {}

    for etype, epattern in EVIDENCE_TYPES.items():
        matches = []
        for pattern in epattern.patterns:
            found = re.findall(pattern, normalized, re.IGNORECASE)
            if found:
                matches.extend(
                    found if isinstance(found[0], str)
                    else [m[0] if isinstance(m, tuple) else m for m in found]
                )

        if matches:
            evidence_types.append(etype)
            evidence_matches[etype] = list(set(matches))[:5]

    return {
        "has_evidence": len(evidence_types) > 0,
        "evidence_types": evidence_types,
        "evidence_matches": evidence_matches,
    }


def extract_kpi_values(text: str) -> list[str]:
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
    print(f"Detecting evidence in {len(df):,} sentences...")

    results = df[text_col].apply(lambda t: detect_evidence(str(t)))

    df = df.copy()
    df["has_evidence"] = results.apply(lambda r: r["has_evidence"])
    df["evidence_types"] = results.apply(lambda r: r["evidence_types"])
    df["kpi_values"] = df[text_col].apply(lambda t: extract_kpi_values(str(t)))

    return df
