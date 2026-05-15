import re
from dataclasses import dataclass
from typing import List

@dataclass
class LabelingRule:
    name: str
    patterns: List[str]
    source: str
    description: str


GRI_ENVIRONMENT_RULES = [
    LabelingRule(
        name="GRI301_Materials",
        patterns=[
            r"\b(nguyên vật liệu|vật liệu tái chế|tái chế|recycled material)\b",
            r"\b(tiêu thụ nguyên liệu|material consumption)\b",
        ],
        source="GRI 301: Materials 2016",
        description="Materials usage and recycling",
    ),
    LabelingRule(
        name="GRI302_Energy",
        patterns=[
            r"\b(năng lượng|energy|kWh|MWh|GWh|MW)\b",
            r"\b(tiêu thụ năng lượng|energy consumption|cường độ năng lượng)\b",
            r"\b(năng lượng tái tạo|renewable energy|điện mặt trời|solar|wind)\b",
            r"\b(tiết kiệm năng lượng|energy saving|energy efficiency)\b",
        ],
        source="GRI 302: Energy 2016",
        description="Energy consumption, intensity, and reduction",
    ),
    LabelingRule(
        name="GRI303_Water",
        patterns=[
            r"\b(nước|water|m3|tài nguyên nước|nguồn nước)\b",
            r"\b(nước thải|wastewater|xử lý nước|water treatment)\b",
        ],
        source="GRI 303: Water and Effluents 2018",
        description="Water withdrawal, consumption, and discharge",
    ),
    LabelingRule(
        name="GRI305_Emissions",
        patterns=[
            r"\b(chuyển đổi xanh|green transition|chuyển dịch năng lượng|energy transition)\b",
            r"\b(kiểm kê khí nhà kính|ghg inventory|phát thải ròng bằng không|tín chỉ carbon|carbon credit)\b",
            r"\b(khí thải|phát thải|emission|CO2|carbon|GHG)\b",
            r"\b(Scope\s*[123]|phạm vi\s*[123])\b",
            r"\b(carbon footprint|dấu chân carbon|khí nhà kính)\b",
            r"\b(net[\-\s]?zero|trung hòa carbon|carbon neutral)\b",
            r"\b(giảm phát thải|emission reduction|carbon offset)\b",
        ],
        source="GRI 305: Emissions 2016",
        description="GHG emissions, reduction, intensity",
    ),
    LabelingRule(
        name="GRI306_Waste",
        patterns=[
            r"\b(kinh tế tuần hoàn|circular economy|rác thải nhựa|plastic waste|vật liệu phân hủy sinh học)\b",
            r"\b(chất thải|rác thải|waste|hazardous waste)\b",
            r"\b(xử lý chất thải|waste management|waste disposal)\b",
            r"\b(ô nhiễm|pollution|contamination)\b",
        ],
        source="GRI 306: Waste 2020",
        description="Waste generation, disposal, recycling",
    ),
    LabelingRule(
        name="GRI304_Biodiversity",
        patterns=[
            r"\b(đa dạng sinh học|biodiversity|hệ sinh thái|ecosystem)\b",
            r"\b(bảo tồn|conservation|rừng|forest|deforestation)\b",
        ],
        source="GRI 304: Biodiversity 2016",
        description="Biodiversity impacts and conservation",
    ),
    LabelingRule(
        name="Climate_Finance",
        patterns=[
            r"\b(khung tài chính xanh|green framework|esg loan|khoản vay xanh|tài trợ dự án xanh)\b",
            r"\b(tín dụng xanh|green credit|trái phiếu xanh|green bond)\b",
            r"\b(tài chính xanh|green finance|tài chính bền vững)\b",
            r"\b(tài chính khí hậu|climate finance)\b",
        ],
        source="GRI 201 + TCFD Guidelines",
        description="Green financial instruments and climate finance",
    ),
]

GRI_SOCIAL_LABOR_RULES = [
    LabelingRule(
        name="GRI401_Employment",
        patterns=[
            r"\b(tuyển dụng|recruitment|nhân sự mới|new hires)\b",
            r"\b(phúc lợi|benefits|chế độ đãi ngộ|compensation)\b",
            r"\b(lương|salary|thu nhập|income|thưởng|bonus)\b",
        ],
        source="GRI 401: Employment 2016",
        description="Employment practices, benefits, retention",
    ),
    LabelingRule(
        name="GRI403_OHS",
        patterns=[
            r"\b(an toàn lao động|occupational health|safety|ATLĐ)\b",
            r"\b(sức khỏe nghề nghiệp|workplace safety)\b",
            r"\b(tai nạn lao động|work injury|incident)\b",
        ],
        source="GRI 403: Occupational Health and Safety 2018",
        description="Workplace health and safety",
    ),
    LabelingRule(
        name="GRI404_Training",
        patterns=[
            r"\b(giữ chân nhân tài|talent retention|đánh giá hiệu quả công việc|performance review|lộ trình thăng tiến)\b",
            r"\b(đào tạo|training|phát triển nhân sự|staff development)\b",
            r"\b(nâng cao năng lực|capacity building|skill)\b",
            r"\b(giờ đào tạo|training hours|chương trình đào tạo)\b",
        ],
        source="GRI 404: Training and Education 2016",
        description="Employee training and development",
    ),
    LabelingRule(
        name="GRI405_Diversity",
        patterns=[
            r"\b(bình đẳng giới|gender equality|diversity)\b",
            # "đa dạng" alone is too common (product variety, etc.); require ESG context
            r"\b(đa dạng hóa nhân sự|lực lượng lao động đa dạng|đa dạng về giới|đa dạng và hòa nhập)\b",
            r"\b(nữ giới|female|phụ nữ|women in leadership)\b",
            r"\b(hòa nhập|inclusion|công bằng|equity)\b",
        ],
        source="GRI 405: Diversity and Equal Opportunity 2016",
        description="Diversity, equity, and inclusion",
    ),
    LabelingRule(
        name="GRI402_LaborRelations",
        patterns=[
            # Requires explicit labor relations context, not just any mention of employees
            r"\b(quan hệ lao động|labor relations|labor.management relations)\b",
            r"\b(công đoàn|union|tổ chức công đoàn|trade union)\b",
            r"\b(thỏa ước lao động|collective agreement|thương lượng tập thể)\b",
            r"\b(môi trường làm việc|work environment|workplace culture)\b",
            r"\b(văn hóa doanh nghiệp|corporate culture|organizational culture)\b",
            r"\b(phúc lợi người lao động|employee wellbeing|phúc lợi CBNV)\b",
        ],
        source="GRI 402: Labor/Management Relations 2016",
        description="Labor relations, collective agreements, work conditions",
    ),
]

GRI_SOCIAL_COMMUNITY_RULES = [
    LabelingRule(
        name="GRI413_Community",
        patterns=[
            r"\b(cộng đồng|community|địa phương|local)\b",
            r"\b(từ thiện|charity|thiện nguyện|volunteer)\b",
            r"\b(trách nhiệm xã hội|social responsibility|CSR)\b",
            r"\b(an sinh xã hội|social welfare)\b",
            r"\b(học bổng|scholarship|quỹ xã hội|social fund)\b",
            r"\b(phát triển cộng đồng|community development)\b",
            r"\b(cứu trợ|relief|hiến máu|blood donation|tài trợ giáo dục|tài trợ y tế)\b",
        ],
        source="GRI 413: Local Communities 2016",
        description="Community engagement and social programs",
    ),
]

GRI_SOCIAL_PRODUCT_RULES = [
    LabelingRule(
        name="GRI416_CustomerHealth",
        patterns=[
            r"\b(bảo vệ.*khách hàng|consumer protection)\b",
            r"\b(quyền lợi khách hàng|customer rights)\b",
        ],
        source="GRI 416: Customer Health and Safety 2016",
        description="Customer health and safety",
    ),
    LabelingRule(
        name="GRI417_Marketing",
        patterns=[
            r"\b(minh bạch thông tin sản phẩm|product labeling)\b",
            r"\b(chất lượng dịch vụ|service quality)\b",
            r"\b(trải nghiệm khách hàng|customer experience)\b",
        ],
        source="GRI 417: Marketing and Labeling 2016",
        description="Product/service marketing and labeling",
    ),
    LabelingRule(
        name="GRI418_Privacy",
        patterns=[
            # "bảo mật" alone triggers on "kênh số ổn định, an toàn, bảo mật"; require data/info context
            r"\b(bảo mật dữ liệu|bảo mật thông tin|data protection|privacy)\b",
            r"\b(bảo vệ dữ liệu|bảo vệ thông tin khách hàng)\b",
            r"\b(an toàn thông tin|information security|an ninh mạng|cybersecurity)\b",
            r"\b(dữ liệu cá nhân|personal data|pdpa|nddp|nghị định 13|chuẩn pci dss)\b",
        ],
        source="GRI 418: Customer Privacy 2016",
        description="Customer data privacy and security",
    ),
    LabelingRule(
        name="FinancialInclusion",
        patterns=[
            r"\b(tài chính toàn diện|financial inclusion)\b",
            r"\b(giáo dục tài chính|financial literacy)\b",
            r"\b(tiếp cận tài chính|access to finance)\b",
        ],
        source="SDG 8: Decent Work, SDG 10: Reduced Inequality",
        description="Financial inclusion and literacy",
    ),
]

GRI_GOVERNANCE_RULES = [
    LabelingRule(
        name="GRI205_AntiCorruption",
        patterns=[
            r"\b(chống tham nhũng|anti-corruption|liêm chính|integrity)\b",
            r"\b(đạo đức kinh doanh|business ethics|code of conduct)\b",
            r"\b(phòng chống rửa tiền|aml|biết khách hàng là ai|kyc|chống tài trợ khủng bố|cft)\b",
            r"\b(bảo vệ người tố giác|whistleblower|xung đột lợi ích|conflict of interest)\b",
        ],
        source="GRI 205: Anti-corruption 2016",
        description="Anti-corruption policies and practices",
    ),
    LabelingRule(
        name="GRI2_Governance",
        patterns=[
            r"\b(quản trị công ty|corporate governance)\b",
            r"\b(hội đồng quản trị|board of directors|HĐQT)\b",
            r"\b(minh bạch|transparency|công bố thông tin|disclosure)\b",
            r"\b(ủy ban esg|esg committee|ban chỉ đạo esg|tích hợp esg|esg integration)\b",
        ],
        source="GRI 2: General Disclosures 2021",
        description="Corporate governance structure",
    ),
    LabelingRule(
        name="RiskManagement",
        patterns=[
            r"\b(quản trị rủi ro|risk management|quản lý rủi ro)\b",
            r"\b(kiểm soát nội bộ|internal control|kiểm toán nội bộ)\b",
            r"\b(tuân thủ|compliance|quy định|regulation)\b",
        ],
        source="GRI 2 + Basel III Framework",
        description="Risk management and compliance",
    ),
    LabelingRule(
        name="Audit_Oversight",
        patterns=[
            r"\b(kiểm toán|audit|giám sát|oversight|supervision)\b",
            r"\b(ban kiểm soát|supervisory board)\b",
        ],
        source="GRI 2: General Disclosures 2021",
        description="Audit and supervisory functions",
    ),
]

IMPLEMENTED_VERBS = LabelingRule(
    name="Bloom_HighLevel_Verbs",
    patterns=[
        r"\b(đã triển khai|đã thực hiện|đã hoàn thành|đã đạt được)\b",
        r"\b(đã giảm|đã tăng|đã tiết kiệm|đã cắt giảm|đã xử lý)\b",
        r"\b(hoàn thành|ghi nhận|đạt được|thực hiện được)\b",
        r"\b(triển khai thành công|vận hành|ứng dụng|áp dụng)\b",
    ],
    source="Anderson & Krathwohl (2001). Bloom's Taxonomy Revised [6]",
    description="High-level cognitive verbs in past tense indicating completed actions",
)

IMPLEMENTED_EVIDENCE = LabelingRule(
    name="Quantitative_Results",
    patterns=[
        r"(?:đã|năm 20\d{2}).{0,50}?\d+\s*(?:%|tỷ|triệu|nghìn|tấn|kWh|MWh|CO2|giờ)",
        r"\d+\s*(%|tỷ|triệu|nghìn|tấn|kWh|MWh).*?(so với|giảm|tăng|đạt)",
        r"\btrong năm (20\d{2})\b",
        r"\bnăm (20\d{2})\b.{0,50}?(đạt|hoàn thành|thực hiện)",
    ],
    source="Florstedt, Fahlbusch & Sontheimer (2025) [4] + GRI 'Quantification Principle'",
    description="Quantitative evidence of past performance",
)

PLANNING_INDICATORS = LabelingRule(
    name="Future_Commitment",
    patterns=[
        r"\b(sẽ|dự kiến|kế hoạch|định hướng|mục tiêu)\b",
        r"\b(hướng tới|phấn đấu|đặt mục tiêu|cam kết.*sẽ)\b",
        r"\b(triển khai trong|thực hiện trong giai đoạn)\b",
        r"(đến năm|vào năm|mục tiêu.*?năm)\s*(2025|2026|2027|2028|2029|2030|2050)\b",
        # "Năm 2025/2030, [bank] triển khai/thực hiện..." — sentence starts with future year
        r"^năm\s*(2025|2026|2027|2028|2029|2030|2050)[,\s].{0,60}(triển khai|thực hiện|áp dụng|tập trung|định hướng|chiến lược)",
        r"\b(net zero|trung hòa carbon).*?(2030|2040|2050)\b",
        r"(lộ trình|roadmap).*?(2025|2030|2050)",
    ],
    source="Florstedt, Fahlbusch & Sontheimer (2025) [4]",
    description="Forward-looking statements with specific targets",
)

HEDGING_INDICATORS = LabelingRule(
    name="Hedging_Vagueness",
    patterns=[
        r"\b(có thể|perhaps|maybe|might)\b",
        r"\b(phần nào|somewhat|to some extent)\b",
        r"\b(tương đối|relatively|fairly)\b",
        r"\b(ngày càng|không ngừng|liên tục|dần dần)\b",
        r"\b(góp phần|đóng góp vào|hỗ trợ)\b",
    ],
    source="Hyland (2005). Metadiscourse [3]; Crismore et al. (1993) [5]",
    description="Hedging markers that reduce commitment certainty",
)

BOOSTING_INDICATORS = LabelingRule(
    name="Boosting_Exaggeration",
    patterns=[
        r"\b(luôn luôn|always|definitely|certainly)\b",
        r"\b(rất|highly|extremely|hoàn toàn|absolutely)\b",
        r"\b(hàng đầu|tiên phong|dẫn đầu|leading|pioneer)\b",
        r"\b(xuất sắc|outstanding|vượt trội|world-class)\b",
    ],
    source="Hyland (2005). Metadiscourse [3]",
    description="Boosting markers that amplify claims without evidence",
)

VAGUE_COMMITMENT = LabelingRule(
    name="Vague_Commitment_Language",
    patterns=[
        r"\b(cam kết|hướng tới|tăng cường|đẩy mạnh|tiếp tục)\b",
        r"\b(chú trọng|quan tâm|ưu tiên|nỗ lực)\b",
        r"\b(phát triển bền vững|trách nhiệm xã hội)\b",
        r"\b(nâng cao nhận thức|nâng cao|cải thiện)\b",
        r"\b(đang nghiên cứu|xem xét|trong quá trình|từng bước)\b",
        r"\b(chung tay|góp sức|kêu gọi|đồng hành cùng)\b",
    ],
    source="Florstedt, Fahlbusch & Sontheimer (2025) [4] — 'cheap talk' indicators",
    description="Vague commitment language without actionable specifics",
)

ALL_TOPIC_RULES = {
    "E": GRI_ENVIRONMENT_RULES,
    "S_labor": GRI_SOCIAL_LABOR_RULES,
    "S_community": GRI_SOCIAL_COMMUNITY_RULES,
    "S_product": GRI_SOCIAL_PRODUCT_RULES,
    "G": GRI_GOVERNANCE_RULES,
}

ALL_ACTION_RULES = {
    "Implemented": [IMPLEMENTED_VERBS, IMPLEMENTED_EVIDENCE],
    "Planning": [PLANNING_INDICATORS],
    "Indeterminate": [HEDGING_INDICATORS, BOOSTING_INDICATORS, VAGUE_COMMITMENT],
}


def match_topic_grounded(text: str, context: str = "", section: str = "") -> tuple[str, float, list[str]]:
    text_lower = text.lower()
    ctx_lower = (f"{context} {text}").lower() if context else text_lower

    scores = {t: 0.0 for t in ALL_TOPIC_RULES}
    matched = {t: [] for t in ALL_TOPIC_RULES}

    for topic, rules in ALL_TOPIC_RULES.items():
        for rule in rules:
            for pattern in rule.patterns:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    scores[topic] += 0.4
                    matched[topic].append(rule.name)
                    break
                elif re.search(pattern, ctx_lower, re.IGNORECASE):
                    scores[topic] += 0.1
                    matched[topic].append(f"{rule.name}(ctx)")
                    break

    best_topic = max(scores, key=scores.get)
    best_score = scores[best_topic]

    if best_score < 0.3:
        return "Non_ESG", 0.5, []

    return best_topic, min(best_score, 1.0), matched[best_topic]


def match_actionability_grounded(text: str, context: str = "") -> tuple[str, float, list[str]]:
    text_lower = text.lower()
    ctx_lower = (f"{context} {text}").lower() if context else text_lower

    scores = {label: 0.0 for label in ALL_ACTION_RULES}
    matched = {label: [] for label in ALL_ACTION_RULES}

    for label, rules in ALL_ACTION_RULES.items():
        for rule in rules:
            for pattern in rule.patterns:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    scores[label] += 0.5
                    matched[label].append(rule.name)
                    break
                elif re.search(pattern, ctx_lower, re.IGNORECASE):
                    scores[label] += 0.2
                    matched[label].append(f"{rule.name}(ctx)")
                    break

    has_numbers = bool(re.search(r"\d+\s*(%|tỷ|triệu|nghìn|tấn|kg|kWh|MWh)", text_lower))
    has_future_year = bool(re.search(r"(2025|2026|2027|2028|2029|2030|2050)", text_lower))

    if has_numbers:
        scores["Indeterminate"] -= 0.3
    if has_future_year:
        scores["Indeterminate"] -= 0.2
        scores["Planning"] += 0.2

    best_label = max(scores, key=scores.get)
    best_score = scores[best_label]

    if best_score < 0.4:
        return "Indeterminate", 0.3, []

    return best_label, min(best_score, 1.0), matched[best_label]


def get_rule_provenance() -> dict:
    """Return all rules with their academic sources. Used for thesis methodology."""
    provenance = {}

    for topic, rules in ALL_TOPIC_RULES.items():
        provenance[topic] = [
            {"name": r.name, "source": r.source, "description": r.description, "num_patterns": len(r.patterns)}
            for r in rules
        ]

    for label, rules in ALL_ACTION_RULES.items():
        provenance[f"Action_{label}"] = [
            {"name": r.name, "source": r.source, "description": r.description, "num_patterns": len(r.patterns)}
            for r in rules
        ]

    return provenance


if __name__ == "__main__":
    prov = get_rule_provenance()
    print("=" * 60)
    print("GROUNDED LABELING RULES SUMMARY")
    print("=" * 60)

    for category, rules in prov.items():
        print(f"\n--- {category} ---")
        for r in rules:
            print(f"  {r['name']:30s}  ({r['num_patterns']} patterns)  Source: {r['source']}")

    print("\n\n" + "=" * 60)
    print("TEST EXAMPLES")
    print("=" * 60)

    tests = [
        "Ngân hàng đã giảm phát thải CO2 được 15% so với năm 2022.",
        "Chúng tôi cam kết hướng tới phát triển bền vững.",
        "Mục tiêu đạt net zero vào năm 2050 theo lộ trình đã đề ra.",
        "Ngân hàng luôn quan tâm, chú trọng đến môi trường làm việc cho CBNV.",
        "Đã triển khai chương trình đào tạo cho 5.000 nhân viên trong năm 2023.",
    ]

    for t in tests:
        topic, t_conf, t_rules = match_topic_grounded(t)
        action, a_conf, a_rules = match_actionability_grounded(t)
        print(f"\n\"{t}\"")
        print(f"  Topic: {topic} (conf={t_conf:.2f}) — {t_rules}")
        print(f"  Action: {action} (conf={a_conf:.2f}) — {a_rules}")
