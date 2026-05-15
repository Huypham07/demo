from __future__ import annotations

import html
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from src.pipeline.ewri import (
    ACTION_PENALTY,
    CONTRADICTION_AMPLIFIER,
    EVIDENCE_SENSITIVITY,
    EWRIScore,
    calculate_topic_entropy,
)
from src.training.labeling.grounded_rules import ALL_ACTION_RULES

TOPIC_LABELS_VI = {
    "E": "Môi trường",
    "S_labor": "Xã hội – Lao động",
    "S_community": "Xã hội – Cộng đồng",
    "S_product": "Xã hội – Sản phẩm",
    "G": "Quản trị",
    "Non_ESG": "Phi ESG",
}

EVIDENCE_TYPES = ["Third_party", "KPI", "Standard", "Time_bound"]

WASHING_CATEGORY_BY_RULE = {
    "Hedging_Vagueness":     "Vagueness (rào đón mơ hồ)",
    "Boosting_Exaggeration": "Cherry-picking (phóng đại)",
    "Vague_Commitment":      "Vague Commitment (cam kết rỗng)",
    "Future_Commitment":     "Decoupling (hứa hẹn chưa thực thi)",
}

# ── HTML template ──────────────────────────────────────────────────────────────

_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: 'Segoe UI', Tahoma, sans-serif;
  background: #f0f2f5;
  color: #222;
  line-height: 1.65;
  padding: 24px 16px;
}
.container { max-width: 1100px; margin: 0 auto; }
h1 {
  font-size: 1.7em;
  color: #1a1a2e;
  border-bottom: 3px solid #e94560;
  padding-bottom: 10px;
  margin-bottom: 6px;
}
.ts { color: #888; font-size: 0.85em; margin-bottom: 24px; }
h2 {
  font-size: 1.2em;
  color: #16213e;
  border-left: 4px solid #0f3460;
  padding-left: 10px;
  margin: 32px 0 14px;
}
h3 { font-size: 1em; color: #1a1a2e; margin-bottom: 8px; }

/* ── Summary cards ── */
.stat-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
  gap: 12px;
  margin-bottom: 10px;
}
.stat-card {
  background: #fff;
  border-radius: 8px;
  padding: 14px 16px;
  box-shadow: 0 1px 4px rgba(0,0,0,.1);
}
.stat-card .val { font-size: 1.7em; font-weight: 700; color: #e94560; line-height: 1.1; }
.stat-card .lbl { font-size: 0.8em; color: #666; margin-top: 2px; }

/* ── Note / callout ── */
.note {
  background: #fff8e1;
  border-left: 4px solid #ffc107;
  padding: 10px 14px;
  border-radius: 0 6px 6px 0;
  font-size: 0.88em;
  color: #555;
  margin: 14px 0;
}

/* ── Tables ── */
.tbl-wrap { overflow-x: auto; margin: 10px 0; }
table {
  width: 100%;
  border-collapse: collapse;
  background: #fff;
  border-radius: 8px;
  overflow: hidden;
  box-shadow: 0 1px 4px rgba(0,0,0,.08);
  font-size: 0.9em;
}
thead th {
  background: #16213e;
  color: #fff;
  padding: 9px 12px;
  text-align: left;
  white-space: nowrap;
}
td {
  padding: 7px 12px;
  border-bottom: 1px solid #f0f0f0;
  vertical-align: top;
}
tr:last-child td { border-bottom: none; }
tr:hover td { background: #f8f9fa; }

/* ── Claim cards ── */
.claim-card {
  background: #fff;
  border-radius: 10px;
  padding: 20px 22px;
  margin: 18px 0;
  box-shadow: 0 2px 8px rgba(0,0,0,.07);
  border-left: 5px solid #dc3545;
}
.claim-card.wrs-mid  { border-left-color: #fd7e14; }
.claim-card.wrs-low  { border-left-color: #ffc107; }
.claim-header { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin-bottom: 14px; }
.badge {
  display: inline-block;
  padding: 3px 9px;
  border-radius: 20px;
  font-size: 0.8em;
  font-weight: 700;
  letter-spacing: .02em;
}
.b-wrs      { background: #e94560; color: #fff; font-size: 0.95em; }
.b-indet    { background: #dc3545; color: #fff; }
.b-planning { background: #fd7e14; color: #fff; }
.b-impl     { background: #28a745; color: #fff; }
.b-topic    { background: #6610f2; color: #fff; }
.b-section  { background: #6c757d; color: #fff; font-size: 0.75em; max-width: 500px;
              overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

/* ── Context block ── */
.ctx-label {
  font-size: 0.75em;
  text-transform: uppercase;
  letter-spacing: .06em;
  color: #aaa;
  margin-bottom: 5px;
}
.ctx-wrapper {
  border: 1px solid #dee2e6;
  border-radius: 6px;
  overflow: hidden;
  margin-bottom: 12px;
}
.ctx-adj {
  background: #f0f0f0;
  padding: 9px 14px;
  font-size: 0.85em;
  color: #888;
  line-height: 1.6;
  white-space: pre-wrap;
  word-break: break-word;
  border-bottom: 1px dashed #ccc;
}
.ctx-adj.ctx-next { border-bottom: none; border-top: 1px dashed #ccc; }
.ctx-adj-label {
  font-size: 0.72em;
  text-transform: uppercase;
  letter-spacing: .05em;
  color: #bbb;
  margin-bottom: 3px;
}
.ctx-block {
  background: #f8f9fa;
  border-left: 3px solid #ced4da;
  padding: 11px 15px;
  font-size: 0.93em;
  white-space: pre-wrap;
  word-break: break-word;
}
.ctx-wrapper .ctx-block { border-left: none; border-left: 3px solid #ced4da; }
mark.hl-sent {
  background: #fff3cd;
  border: 1px solid #ffc107;
  border-radius: 3px;
  padding: 1px 3px;
  font-weight: 700;
}

/* ── Washing reasons ── */
.washing-cats { margin-bottom: 8px; }
.cat-badge {
  display: inline-block;
  background: #fff3cd;
  border: 1px solid #ffc107;
  border-radius: 4px;
  padding: 2px 8px;
  font-size: 0.82em;
  margin: 3px 3px 3px 0;
}
.reason-list { list-style: none; padding: 0; margin: 0; }
.reason-list li {
  padding: 3px 0 3px 18px;
  position: relative;
  font-size: 0.9em;
  color: #444;
}
.reason-list li::before { content: "\\2192"; position: absolute; left: 0; color: #aaa; }

/* ── Evidence block ── */
.ev-block {
  background: #e8f4f8;
  border: 1px solid #bee5eb;
  border-radius: 6px;
  padding: 11px 14px;
  margin-top: 12px;
}
.ev-header {
  font-weight: 700;
  color: #0c5460;
  font-size: 0.85em;
  margin-bottom: 6px;
  display: flex;
  align-items: center;
  gap: 8px;
}
.nli { display: inline-block; padding: 1px 7px; border-radius: 10px; font-size: 0.8em; font-weight: 700; }
.nli-contradiction { background: #f8d7da; color: #721c24; }
.nli-entailment    { background: #d4edda; color: #155724; }
.nli-neutral       { background: #e2e3e5; color: #383d41; }
.ev-text { color: #0c5460; font-size: 0.9em; white-space: pre-wrap; word-break: break-word; }

/* ── Positive cards (smaller) ── */
.pos-card {
  background: #fff;
  border-left: 5px solid #28a745;
  border-radius: 8px;
  padding: 14px 18px;
  margin: 12px 0;
  box-shadow: 0 1px 4px rgba(0,0,0,.07);
  font-size: 0.9em;
}
.pos-card .pos-sent { font-style: italic; color: #333; margin: 6px 0; }
.pos-card .pos-ev   { color: #0c5460; font-size: 0.85em; }
"""

_HTML_HEAD = """<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Báo cáo ESG-Washing &mdash; {bank} {year}</title>
<style>{css}</style>
</head>
<body>
<div class="container">
"""

_HTML_FOOT = "</div></body></html>\n"


# ── Helper utilities ───────────────────────────────────────────────────────────

def _h(s: str) -> str:
    """HTML-escape a string."""
    return html.escape(str(s))

def _fmt_pct(part: float, total: float) -> str:
    return f"{(100 * part / total):.1f}%" if total > 0 else "0.0%"

def _badge(text: str, cls: str) -> str:
    return f'<span class="badge {cls}">{_h(text)}</span>'

def _nli_badge(label: str) -> str:
    cls = {"contradiction": "nli-contradiction", "entailment": "nli-entailment"}.get(label, "nli-neutral")
    return f'<span class="nli {cls}">{_h(label or "neutral")}</span>'

def _table_html(headers: list[str], rows: list[list]) -> str:
    ths = "".join(f"<th>{_h(h)}</th>" for h in headers)
    trs = ""
    for row in rows:
        tds = "".join(f"<td>{_h(str(v))}</td>" for v in row)
        trs += f"<tr>{tds}</tr>\n"
    return f'<div class="tbl-wrap"><table><thead><tr>{ths}</tr></thead><tbody>{trs}</tbody></table></div>'

def _ctx_with_neighbors(block_text: str, sentence: str,
                        prev_text: str = "", next_text: str = "") -> str:
    """Render the three-block context: prev (gray) + current (highlighted) + next (gray)."""
    has_prev = bool(prev_text and prev_text.strip() and prev_text.strip() != block_text.strip())
    has_next = bool(next_text and next_text.strip() and next_text.strip() != block_text.strip())

    if not has_prev and not has_next:
        # No neighbours — render plain block (existing style)
        return f'<div class="ctx-block">{_highlight_sentence_in_block(block_text, sentence)}</div>'

    parts = ['<div class="ctx-wrapper">']
    if has_prev:
        parts.append(f'<div class="ctx-adj">'
                     f'<div class="ctx-adj-label">&#9650; Đoạn trước</div>'
                     f'{_h(prev_text.strip())}</div>')
    parts.append(f'<div class="ctx-block">{_highlight_sentence_in_block(block_text, sentence)}</div>')
    if has_next:
        parts.append(f'<div class="ctx-adj ctx-next">'
                     f'<div class="ctx-adj-label">&#9660; Đoạn sau</div>'
                     f'{_h(next_text.strip())}</div>')
    parts.append('</div>')
    return "".join(parts)


def _highlight_sentence_in_block(block_text: str, sentence: str) -> str:
    """Return HTML of block_text with the sentence wrapped in a highlight mark."""
    safe_block = _h(block_text)
    safe_sent  = _h(sentence)
    mark_open  = '<mark class="hl-sent">'
    mark_close = '</mark>'

    # 1. Try exact match on the escaped strings (most common case)
    idx = safe_block.find(safe_sent)
    if idx >= 0:
        return safe_block[:idx] + mark_open + safe_sent + mark_close + safe_block[idx + len(safe_sent):]

    # 2. Try on raw strings (handles cases where escape changed char codes)
    idx2 = block_text.find(sentence)
    if idx2 >= 0:
        return (_h(block_text[:idx2])
                + mark_open + safe_sent + mark_close
                + _h(block_text[idx2 + len(sentence):]))

    # 3. Fallback: show block, then annotate sentence separately
    return safe_block + f'<br>{mark_open}↑ câu được xét: {safe_sent}{mark_close}'


def _section_breakdown(esg_df: pd.DataFrame, top_n: int = 10) -> list[list]:
    if "section_title" not in esg_df.columns:
        return []
    rows = []
    for section, group in esg_df.groupby("section_title"):
        n = len(group)
        if n < 3:
            continue
        ewri  = float(group["wrs"].mean()) * 100
        impl  = int((group["action_label"] == "Implemented").sum())
        plan  = int((group["action_label"] == "Planning").sum())
        indet = int((group["action_label"] == "Indeterminate").sum())
        ev    = int(group["has_evidence"].sum()) if "has_evidence" in group.columns else 0
        rows.append([str(section)[:70], n, f"{ewri:.1f}",
                     _fmt_pct(indet, n), _fmt_pct(plan, n), _fmt_pct(impl, n), _fmt_pct(ev, n)])
    rows.sort(key=lambda r: float(r[2]), reverse=True)
    return rows[:top_n]


def _explain_washing(
    sentence: str,
    action_label: str,
    has_evidence: bool,
    nli_label: str,
    context: str = "",
) -> dict:
    text_lower = sentence.lower()
    ctx_lower  = (f"{context} {sentence}").lower() if context else text_lower

    matched: list[dict] = []
    for label_target, rules in ALL_ACTION_RULES.items():
        for rule in rules:
            kws: list[str] = []
            for pattern in rule.patterns:
                hits = re.findall(pattern, ctx_lower, re.IGNORECASE)
                for h_ in hits:
                    kw = h_ if isinstance(h_, str) else (h_[0] if h_ else "")
                    if kw and kw not in kws:
                        kws.append(kw)
            if kws:
                matched.append({"rule": rule.name, "label_target": label_target, "keywords": kws[:5]})

    categories: list[str] = []
    for m in matched:
        cat = WASHING_CATEGORY_BY_RULE.get(m["rule"])
        if cat and cat not in categories:
            categories.append(cat)

    reasons: list[str] = []
    if action_label == "Indeterminate" and not categories:
        categories.append("General Vagueness (mơ hồ chung)")
        reasons.append("Câu thiếu động từ hành động cụ thể, KPI, hoặc mốc thời gian.")
    if action_label == "Planning" and not has_evidence:
        if "Decoupling (hứa hẹn chưa thực thi)" not in categories:
            categories.append("Decoupling (hứa hẹn chưa thực thi)")
        reasons.append("Là cam kết tương lai (Planning) nhưng không tìm thấy bằng chứng đi kèm.")
    if nli_label == "contradiction":
        categories.append("Contradiction (mâu thuẫn với bằng chứng)")
        reasons.append("Bằng chứng nội tại của báo cáo phản bác chính tuyên bố này.")
    if not has_evidence and action_label != "Implemented":
        reasons.append("Không có bằng chứng (KPI / chứng nhận / mốc thời gian) đi kèm.")
    for m in matched:
        cat = WASHING_CATEGORY_BY_RULE.get(m["rule"])
        if cat:
            kws_html = ", ".join(f"<code>{_h(k)}</code>" for k in m["keywords"])
            reasons.append(f'{cat}: từ khoá khớp {kws_html} (rule {_h(m["rule"])}).')

    return {
        "washing_categories": categories or ["—"],
        "reasons": reasons or ["Không có dấu hiệu washing rõ ràng."],
    }


# ── Main report generator ──────────────────────────────────────────────────────

def generate_demo_report(
    sentences_df: pd.DataFrame,
    esg_df: pd.DataFrame,
    ewri_score: EWRIScore,
    output_dir: str | Path,
    bank: str,
    year: int,
    metadata: Optional[dict] = None,
) -> Path:
    """Render an HTML + JSON report for one document. Returns the HTML path."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata       = metadata or {}
    total_sentences = len(sentences_df)
    n_esg   = len(esg_df)
    impl    = int((esg_df["action_label"] == "Implemented").sum())
    plan    = int((esg_df["action_label"] == "Planning").sum())
    indet   = int((esg_df["action_label"] == "Indeterminate").sum())
    n_evidence = int(esg_df["has_evidence"].sum()) if "has_evidence" in esg_df.columns else 0

    topic_counts = {t: int((esg_df["topic_label"] == t).sum())
                    for t in ["E", "S_labor", "S_community", "S_product", "G"]}
    entropy = calculate_topic_entropy(topic_counts)

    sent_risks = sorted(ewri_score.sentence_risks, key=lambda r: r.get("washing_risk", 0), reverse=True)
    top_risk    = sent_risks[:50]
    positives   = [r for r in sent_risks if r.get("action_label") == "Implemented" and r.get("has_evidence")]
    positives.sort(key=lambda r: r.get("washing_risk", 0))
    top_positive = positives[:20]

    # block_text lookups from esg_df (fallback for sentence_risks without block_text)
    bt_lookup:      dict[str, str] = {}
    bt_prev_lookup: dict[str, str] = {}
    bt_next_lookup: dict[str, str] = {}
    if "sent_id" in esg_df.columns:
        for _, row in esg_df.iterrows():
            sid = str(row.get("sent_id", "") or "")
            if not sid:
                continue
            bt_lookup[sid]      = str(row.get("block_text", "")      or "")
            bt_prev_lookup[sid] = str(row.get("block_prev_text", "") or "")
            bt_next_lookup[sid] = str(row.get("block_next_text", "") or "")

    # ── Build HTML ─────────────────────────────────────────────────────────────
    parts: list[str] = []
    parts.append(_HTML_HEAD.format(bank=_h(bank), year=year, css=_CSS))
    parts.append(f'<h1>Báo cáo phân tích ESG-Washing &mdash; {_h(bank)} {year}</h1>')

    # 1. Summary
    parts.append('<h2>1. Tóm tắt</h2>')
    char_count = metadata.get("char_count")
    stats = [
        (f"{ewri_score.ewri:.2f} / 100", "EWRI (điểm thô)"),
        (f"{total_sentences:,}", "Tổng số câu"),
        (f"{n_esg:,} ({_fmt_pct(n_esg, total_sentences)})", "Câu ESG"),
        (_fmt_pct(n_evidence, n_esg), "Tỷ lệ có evidence"),
        (f"{entropy:.3f}", "Topic entropy"),
    ]
    parts.append('<div class="stat-grid">')
    for val, lbl in stats:
        parts.append(f'<div class="stat-card"><div class="val">{_h(str(val))}</div>'
                     f'<div class="lbl">{_h(lbl)}</div></div>')
    parts.append('</div>')

    # 2. EWRI decomposition
    parts.append('<h2>2. Phân rã EWRI theo nhãn hành động</h2>')
    ewri_val = max(ewri_score.ewri, 1e-9)
    parts.append(_table_html(
        ["Nhãn", "Số câu", "Tỷ lệ", "Đóng góp WRS", "% EWRI"],
        [
            ["Indeterminate", indet, _fmt_pct(indet, n_esg),
             f"{ewri_score.contribution_indeterminate:.2f}",
             _fmt_pct(ewri_score.contribution_indeterminate, ewri_val)],
            ["Planning", plan, _fmt_pct(plan, n_esg),
             f"{ewri_score.contribution_planning:.2f}",
             _fmt_pct(ewri_score.contribution_planning, ewri_val)],
            ["Implemented", impl, _fmt_pct(impl, n_esg),
             f"{ewri_score.contribution_implemented:.2f}",
             _fmt_pct(ewri_score.contribution_implemented, ewri_val)],
        ],
    ))

    # 3. Topic distribution
    parts.append('<h2>3. Phân phối chủ đề ESG</h2>')
    t_rows = []
    for topic in ["E", "S_labor", "S_community", "S_product", "G"]:
        sub = esg_df[esg_df["topic_label"] == topic]
        if len(sub) == 0:
            continue
        t_rows.append([
            f"{topic} – {TOPIC_LABELS_VI.get(topic, topic)}",
            len(sub), _fmt_pct(len(sub), n_esg),
            _fmt_pct((sub["action_label"] == "Implemented").sum(), len(sub)),
            _fmt_pct((sub["action_label"] == "Planning").sum(), len(sub)),
            _fmt_pct((sub["action_label"] == "Indeterminate").sum(), len(sub)),
            _fmt_pct(sub["has_evidence"].sum() if "has_evidence" in sub.columns else 0, len(sub)),
        ])
    parts.append(_table_html(
        ["Chủ đề", "Số câu", "Tỷ lệ", "Implemented", "Planning", "Indeterminate", "Có evidence"],
        t_rows,
    ))

    # 4. Evidence analysis
    parts.append('<h2>4. Phân tích bằng chứng (evidence linking)</h2>')
    parts.append(f'<p>Tỷ lệ câu có evidence: <strong>{_fmt_pct(n_evidence, n_esg)}</strong></p>')
    if "evidence_types" in esg_df.columns:
        counts: dict[str, int] = {t: 0 for t in EVIDENCE_TYPES}
        for et in esg_df["evidence_types"]:
            try:
                for t in (list(et) if et is not None else []):
                    if t in counts:
                        counts[t] += 1
            except TypeError:
                pass
        parts.append('<h3 style="margin-top:12px">Loại bằng chứng (theo phân cấp GRI)</h3>')
        parts.append(_table_html(
            ["Loại bằng chứng", "Số câu chứa", "Tỷ lệ"],
            [[t, counts[t], _fmt_pct(counts[t], n_esg)] for t in EVIDENCE_TYPES],
        ))
    if "nli_label" in esg_df.columns:
        has_ev_df = esg_df[esg_df["has_evidence"].astype(bool)] if "has_evidence" in esg_df.columns else esg_df
        if len(has_ev_df):
            nli_counts = has_ev_df["nli_label"].value_counts(dropna=False)
            parts.append('<h3 style="margin-top:12px">Phán định NLI (mDeBERTa-v3-base-xnli)</h3>')
            parts.append(_table_html(
                ["NLI label", "Số câu", "Tỷ lệ"],
                [[str(k or "neutral"), int(v), _fmt_pct(v, len(has_ev_df))]
                 for k, v in nli_counts.items()],
            ))

    # 5. Top-50 risk claims
    parts.append('<h2>5. Top 50 câu rủi ro cao nhất (kèm lý do flagged)</h2>')
    parts.append('<p style="margin-bottom:12px;font-size:.9em;color:#555">'
                 'Mỗi câu được phân tích bằng quy tắc ký hiệu (Bloom + Hyland). '
                 'Câu được xét <mark class="hl-sent">nổi bật màu vàng</mark> trong đoạn văn gốc.</p>')

    explained_top: list[dict] = []
    for rank, r in enumerate(top_risk):
        sentence    = str(r.get("sentence", ""))
        sent_id     = str(r.get("sent_id", ""))
        action_lbl  = str(r.get("action_label", "?"))
        topic       = str(r.get("topic", "?"))
        wrs         = float(r.get("washing_risk", 0))
        section_ttl = str(r.get("section_title", "") or "")
        nli_lbl     = str(r.get("nli_label", "") or "")
        best_ev     = str(r.get("best_evidence", "") or "").strip()
        # block_text: prefer what's stored in sent_risks, fall back to lookup
        block_text  = str(r.get("block_text", "") or bt_lookup.get(sent_id, "")).strip()

        explanation = _explain_washing(
            sentence=sentence, action_label=action_lbl,
            has_evidence=bool(r.get("has_evidence", False)), nli_label=nli_lbl,
        )

        card_cls = "claim-card" + (" wrs-mid" if 0.5 <= wrs < 0.7 else "") + (" wrs-low" if wrs < 0.5 else "")
        act_cls  = {"Indeterminate": "b-indet", "Planning": "b-planning", "Implemented": "b-impl"}.get(action_lbl, "b-indet")

        parts.append(f'<div class="{card_cls}">')
        parts.append('<div class="claim-header">')
        parts.append(f'<span style="font-weight:700;color:#666">#{rank+1}</span>')
        if section_ttl and section_ttl != "UNKNOWN":
            parts.append(_badge(f"{section_ttl}", "b-section"))
        parts.append(_badge(f"WRS = {wrs:.3f}", "b-wrs"))
        parts.append(_badge(action_lbl, act_cls))
        parts.append(_badge(f"{topic} – {TOPIC_LABELS_VI.get(topic, topic)}", "b-topic"))
        if section_ttl and section_ttl != "UNKNOWN":
            parts.append(_badge(f"{section_ttl}", "b-section"))
        parts.append('</div>')

        # Context — current block + adjacent blocks from lookup
        block_prev = str(r.get("block_prev_text", "") or bt_prev_lookup.get(sent_id, "")).strip()
        block_next = str(r.get("block_next_text", "") or bt_next_lookup.get(sent_id, "")).strip()
        parts.append('<div class="ctx-label">Ngữ cảnh (đoạn trước / đoạn xét / đoạn sau)</div>')
        if block_text:
            parts.append(_ctx_with_neighbors(block_text, sentence, block_prev, block_next))
        else:
            parts.append(f'<div class="ctx-block"><mark class="hl-sent">{_h(sentence)}</mark></div>')

        # Washing categories
        cats_html = "".join(f'<span class="cat-badge">{_h(c)}</span>'
                            for c in explanation["washing_categories"])
        parts.append(f'<div class="washing-cats"><strong>Loại washing:</strong> {cats_html}</div>')

        # Reasons
        reasons_html = "".join(f'<li>{reason}</li>' for reason in explanation["reasons"])
        parts.append(f'<ul class="reason-list">{reasons_html}</ul>')

        # Evidence
        if best_ev:
            parts.append('<div class="ev-block">')
            parts.append(f'<div class="ev-header">Bằng chứng liên kết {_nli_badge(nli_lbl)}</div>')
            parts.append(f'<div class="ev-text">{_h(best_ev)}</div>')
            parts.append('</div>')

        parts.append('</div>')
        explained_top.append({**r, "explanation": explanation})

    # 6. Positive examples
    parts.append('<h2>6. Top 20 ví dụ tích cực (Implemented + có evidence)</h2>')
    if top_positive:
        for i, r in enumerate(top_positive):
            sent_id    = str(r.get("sent_id", ""))
            sentence   = str(r.get("sentence", ""))
            block_text = str(r.get("block_text", "") or bt_lookup.get(sent_id, "")).strip()
            block_prev = str(r.get("block_prev_text", "") or bt_prev_lookup.get(sent_id, "")).strip()
            block_next = str(r.get("block_next_text", "") or bt_next_lookup.get(sent_id, "")).strip()
            topic      = str(r.get("topic", "?"))
            best_ev    = str(r.get("best_evidence", "") or "")
            wrs        = float(r.get("washing_risk", 0))

            parts.append('<div class="pos-card">')
            parts.append('<div class="claim-header">')
            parts.append(f'<span style="font-weight:700;color:#666">#{i+1}</span>')
            parts.append(_badge(f"WRS = {wrs:.3f}", "b-impl"))
            parts.append(_badge(topic, "b-topic"))
            parts.append('</div>')

            if block_text:
                parts.append(_ctx_with_neighbors(block_text, sentence, block_prev, block_next))
            else:
                parts.append(f'<div class="pos-sent">{_h(sentence)}</div>')

            if best_ev:
                parts.append(f'<div class="pos-ev"><strong>Bằng chứng:</strong> {_h(best_ev)}</div>')
            parts.append('</div>')
    else:
        parts.append('<p><em>(không có câu nào)</em></p>')

    # 7. Section breakdown
    parts.append('<h2>7. Phân tích theo chương / mục báo cáo</h2>')
    sec_rows = _section_breakdown(esg_df, top_n=50)
    if sec_rows:
        parts.append(_table_html(
            ["Chương / Mục", "Số câu ESG", "EWRI", "Indet %", "Plan %", "Impl %", "Evidence %"],
            sec_rows,
        ))
    else:
        parts.append('<p><em>(không có dữ liệu section)</em></p>')

    parts.append(_HTML_FOOT)

    html_path = output_dir / "report.html"
    html_path.write_text("".join(parts), encoding="utf-8")

    # JSON sidecar
    json_payload = {
        "bank": bank, "year": year, "metadata": metadata,
        "summary": {
            "total_sentences": total_sentences, "esg_sentences": n_esg,
            "ewri_raw": ewri_score.ewri, "topic_entropy": entropy,
        },
        "decomposition": {
            "indeterminate": {"count": indet, "contribution": ewri_score.contribution_indeterminate},
            "planning":      {"count": plan,  "contribution": ewri_score.contribution_planning},
            "implemented":   {"count": impl,  "contribution": ewri_score.contribution_implemented},
        },
        "evidence": {"with_evidence": n_evidence, "rate": n_evidence / n_esg if n_esg else 0.0},
        "top_risk_claims": explained_top if explained_top else top_risk,
        "positive_examples": top_positive,
        "section_breakdown": sec_rows,
        "parameters": {
            "action_penalty": ACTION_PENALTY,
            "evidence_sensitivity": EVIDENCE_SENSITIVITY,
            "contradiction_amplifier": CONTRADICTION_AMPLIFIER,
        },
    }
    (output_dir / "report.json").write_text(
        json.dumps(json_payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    return html_path
