from __future__ import annotations

import html
import json
import re
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

TOPIC_LABELS_VI = {
    "E": "Môi trường",
    "S_labor": "Xã hội – Lao động",
    "S_community": "Xã hội – Cộng đồng",
    "S_product": "Xã hội – Sản phẩm",
    "G": "Quản trị",
    "Non_ESG": "Phi ESG",
}

EVIDENCE_TYPES = ["Third_party", "KPI", "Standard", "Time_bound"]

# ── CSS ───────────────────────────────────────────────────────────────────────

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

/* ── Badge ── */
.badge {
  display: inline-block;
  padding: 3px 9px;
  border-radius: 20px;
  font-size: 0.8em;
  font-weight: 700;
}
.b-indet    { background: #dc3545; color: #fff; }
.b-planning { background: #fd7e14; color: #fff; }
.b-impl     { background: #28a745; color: #fff; }
.b-topic    { background: #6610f2; color: #fff; }

/* ── Document view ── */
.doc-legend {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  align-items: center;
  background: #fff;
  border-radius: 8px;
  padding: 12px 16px;
  margin-bottom: 16px;
  box-shadow: 0 1px 4px rgba(0,0,0,.07);
  font-size: 0.85em;
}
.doc-legend strong { margin-right: 6px; color: #444; }
.leg-swatch {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  padding: 3px 10px;
  border-radius: 4px;
  cursor: default;
  font-size: 0.85em;
}
.leg-impl  { background: rgba(40,167,69,.15); border-bottom: 2px solid #28a745; }
.leg-plan  { background: rgba(253,126,20,.15); border-bottom: 2px solid #fd7e14; }
.leg-indet { background: rgba(220,53,69,.12); border-bottom: 2px solid #dc3545; }
.leg-contr { background: rgba(114,28,36,.18); border-bottom: 2px dashed #721c24; }

.doc-body {
  background: #fff;
  border-radius: 10px;
  padding: 24px 28px;
  box-shadow: 0 1px 6px rgba(0,0,0,.08);
  line-height: 1.85;
  font-size: 0.95em;
}
.doc-section-title {
  font-size: 1.05em;
  font-weight: 700;
  color: #16213e;
  border-left: 3px solid #0f3460;
  padding: 6px 12px;
  margin: 28px 0 12px;
  background: #f4f6fb;
  border-radius: 0 4px 4px 0;
}
.doc-block {
  margin-bottom: 12px;
}

/* ── ESG sentence highlights ── */
.esg-sent {
  cursor: pointer;
  border-radius: 3px;
  padding: 1px 2px;
  transition: filter .15s;
  position: relative;
}
.esg-sent:hover { filter: brightness(0.88); }
.esg-impl  { background: rgba(40,167,69,.15); border-bottom: 2px solid #28a745; }
.esg-plan  { background: rgba(253,126,20,.15); border-bottom: 2px solid #fd7e14; }
.esg-indet { background: rgba(220,53,69,.12); border-bottom: 2px solid #dc3545; }
.esg-contr { background: rgba(114,28,36,.18); border-bottom: 2px dashed #721c24; }
.esg-sent.active { outline: 2px solid #0f3460; outline-offset: 1px; }

/* ── Side panel ── */
.esg-panel {
  position: fixed;
  top: 0;
  right: 0;
  width: 400px;
  max-width: 95vw;
  height: 100vh;
  background: #fff;
  box-shadow: -6px 0 24px rgba(0,0,0,.18);
  overflow-y: auto;
  padding: 0;
  z-index: 2000;
  transform: translateX(0);
  transition: transform .22s cubic-bezier(.4,0,.2,1);
}
.esg-panel.hidden { transform: translateX(110%); }

.panel-top {
  position: sticky;
  top: 0;
  background: #16213e;
  color: #fff;
  padding: 14px 18px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  z-index: 10;
}
.panel-top h4 { font-size: 0.95em; margin: 0; opacity: 0.85; }
.panel-close {
  background: none;
  border: none;
  color: #fff;
  font-size: 1.4em;
  cursor: pointer;
  line-height: 1;
  opacity: 0.75;
  padding: 0 4px;
}
.panel-close:hover { opacity: 1; }

.panel-body { padding: 18px; }

.panel-sentence {
  font-size: 0.93em;
  color: #222;
  line-height: 1.7;
  padding: 12px 14px;
  background: #f8f9fa;
  border-radius: 6px;
  border-left: 3px solid #ccc;
  margin-bottom: 16px;
  word-break: break-word;
}

.panel-badges { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 14px; }
.panel-badge {
  display: inline-block;
  padding: 3px 10px;
  border-radius: 20px;
  font-size: 0.82em;
  font-weight: 700;
}

.panel-stat-row {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
  margin-bottom: 14px;
}
.panel-stat {
  background: #f4f6fb;
  border-radius: 6px;
  padding: 8px 12px;
  font-size: 0.85em;
}
.panel-stat .ps-label { color: #888; font-size: 0.82em; margin-bottom: 2px; }
.panel-stat .ps-value { font-weight: 700; color: #222; }

.panel-ev-box {
  background: #e8f4f8;
  border: 1px solid #bee5eb;
  border-radius: 8px;
  padding: 12px 14px;
  margin-top: 4px;
}
.panel-ev-label {
  font-size: 0.8em;
  text-transform: uppercase;
  letter-spacing: .05em;
  color: #0c5460;
  font-weight: 700;
  margin-bottom: 8px;
  display: flex;
  align-items: center;
  gap: 8px;
}
.panel-ev-text {
  font-size: 0.9em;
  color: #0c5460;
  line-height: 1.65;
  word-break: break-word;
  white-space: pre-wrap;
}
.panel-no-ev {
  color: #aaa;
  font-size: 0.88em;
  font-style: italic;
  margin-top: 4px;
}

.nli-pill {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 10px;
  font-size: 0.8em;
  font-weight: 700;
}
.nli-entailment    { background: #d4edda; color: #155724; }
.nli-contradiction { background: #f8d7da; color: #721c24; }
.nli-neutral       { background: #e2e3e5; color: #383d41; }

/* Backdrop */
.panel-backdrop {
  position: fixed;
  inset: 0;
  background: rgba(0,0,0,.2);
  z-index: 1999;
  display: none;
}
.panel-backdrop.visible { display: block; }
"""

# ── JavaScript ────────────────────────────────────────────────────────────────

_JS_TEMPLATE = """
(function() {
  var ESG_DATA = %s;

  var panel    = document.getElementById('esg-panel');
  var backdrop = document.getElementById('panel-backdrop');
  var pbody    = document.getElementById('esg-panel-body');
  var activeEl = null;

  function openPanel(sentEl) {
    var id = sentEl.dataset.id;
    var d  = ESG_DATA[id];
    if (!d) return;
    if (activeEl) activeEl.classList.remove('active');
    activeEl = sentEl;
    sentEl.classList.add('active');
    pbody.innerHTML = buildPanelHTML(d);
    panel.classList.remove('hidden');
    backdrop.classList.add('visible');
  }

  function closePanel() {
    panel.classList.add('hidden');
    backdrop.classList.remove('visible');
    if (activeEl) { activeEl.classList.remove('active'); activeEl = null; }
  }

  document.getElementById('panel-close-btn').addEventListener('click', closePanel);
  backdrop.addEventListener('click', closePanel);

  document.addEventListener('click', function(e) {
    var el = e.target.closest('.esg-sent');
    if (el) { openPanel(el); return; }
    if (!panel.contains(e.target) && !backdrop.contains(e.target)) {
      if (!panel.classList.contains('hidden')) closePanel();
    }
  });

  var TOPIC_VI = {
    E: 'Môi trường', S_labor: 'Xã hội – Lao động',
    S_community: 'Xã hội – Cộng đồng', S_product: 'Xã hội – Sản phẩm', G: 'Quản trị'
  };
  var ACTION_COLOR = { Implemented: '#28a745', Planning: '#fd7e14', Indeterminate: '#dc3545' };
  var NLI_CLS = { entailment: 'nli-entailment', contradiction: 'nli-contradiction', neutral: 'nli-neutral' };

  function esc(s) {
    return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
  }

  function buildPanelHTML(d) {
    var ac = ACTION_COLOR[d.action] || '#666';
    var nc = NLI_CLS[d.nli_label] || 'nli-neutral';
    var wrsColor = d.wrs >= 0.5 ? '#dc3545' : d.wrs >= 0.3 ? '#fd7e14' : '#28a745';
    var topicStr = d.topic + (TOPIC_VI[d.topic] ? ' – ' + TOPIC_VI[d.topic] : '');

    var html = '';

    // Sentence
    html += '<div class="panel-sentence">' + esc(d.sentence) + '</div>';

    // Badges
    html += '<div class="panel-badges">';
    html += '<span class="panel-badge" style="background:' + ac + ';color:#fff">' + esc(d.action) + '</span>';
    html += '<span class="panel-badge" style="background:#6610f2;color:#fff">' + esc(topicStr) + '</span>';
    html += '<span class="panel-badge" style="background:' + wrsColor + ';color:#fff">WRS = ' + d.wrs.toFixed(3) + '</span>';
    html += '</div>';

    // Stats grid
    html += '<div class="panel-stat-row">';
    html += '<div class="panel-stat"><div class="ps-label">NLI label</div>';
    html += '<span class="nli-pill ' + nc + '">' + esc(d.nli_label || '–') + '</span></div>';
    html += '<div class="panel-stat"><div class="ps-label">NLI score</div>';
    html += '<div class="ps-value">' + (d.nli_score||0).toFixed(3) + '</div></div>';
    html += '<div class="panel-stat"><div class="ps-label">Similarity</div>';
    html += '<div class="ps-value">' + (d.similarity||0).toFixed(3) + '</div></div>';
    html += '<div class="panel-stat"><div class="ps-label">Topic conf.</div>';
    html += '<div class="ps-value">' + (d.topic_conf||0).toFixed(3) + '</div></div>';
    html += '</div>';

    // Evidence
    html += '<div class="panel-ev-box">';
    html += '<div class="panel-ev-label">Bằng chứng li\xean kết <span class="nli-pill ' + nc + '">' + esc(d.nli_label||'neutral') + '</span></div>';
    if (d.evidence) {
      html += '<div class="panel-ev-text">' + esc(d.evidence) + '</div>';
    } else {
      html += '<div class="panel-no-ev">Kh\xf4ng t\xecm được bằng chứng x\xe1c nhận.</div>';
    }
    html += '</div>';

    return html;
  }
})();
"""

# ── HTML shell ────────────────────────────────────────────────────────────────

_HTML_HEAD = """<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Báo cáo ESG-Washing &mdash; {bank} {year}</title>
<style>{css}</style>
</head>
<body>
<div id="panel-backdrop" class="panel-backdrop"></div>
<div id="esg-panel" class="esg-panel hidden">
  <div class="panel-top">
    <h4>Chi tiết c\xe2u ESG</h4>
    <button id="panel-close-btn" class="panel-close" title="Đ\xf3ng">&times;</button>
  </div>
  <div class="panel-body" id="esg-panel-body">
    <p style="color:#aaa;font-size:.9em">(Click v\xe0o một c\xe2u ESG được t\xf4 m\xe0u trong t\xe0i liệu để xem chi tiết.)</p>
  </div>
</div>
<div class="container">
"""

_HTML_FOOT = "</div></body></html>\n"

# ── Helper utilities ──────────────────────────────────────────────────────────

def _h(s: str) -> str:
    return html.escape(str(s))

def _fmt_pct(part: float, total: float) -> str:
    return f"{(100 * part / total):.1f}%" if total > 0 else "0.0%"

def _table_html(headers: list[str], rows: list[list]) -> str:
    ths = "".join(f"<th>{_h(h)}</th>" for h in headers)
    trs = ""
    for row in rows:
        tds = "".join(f"<td>{_h(str(v))}</td>" for v in row)
        trs += f"<tr>{tds}</tr>\n"
    return f'<div class="tbl-wrap"><table><thead><tr>{ths}</tr></thead><tbody>{trs}</tbody></table></div>'

def _nli_pill(label: str) -> str:
    cls = {"contradiction": "nli-contradiction", "entailment": "nli-entailment"}.get(label, "nli-neutral")
    return f'<span class="nli-pill {cls}">{_h(label or "neutral")}</span>'

def _sort_key(sent_id: str):
    m = re.search(r's(\d+)_(\d+)', str(sent_id))
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)

# ── Document view ─────────────────────────────────────────────────────────────

def _build_popup_data(esg_df: pd.DataFrame) -> dict:
    """Build {sent_id: info_dict} for JS popup."""
    result = {}
    for _, row in esg_df.iterrows():
        sid = str(row.get("sent_id", "") or "")
        if not sid:
            continue
        nli_lbl = str(row.get("nli_label", "") or "neutral")
        result[sid] = {
            "sentence":   str(row.get("sentence", "") or ""),
            "action":     str(row.get("action_label", "") or ""),
            "topic":      str(row.get("topic_label", "") or ""),
            "wrs":        float(row.get("wrs", 0) or 0),
            "nli_label":  nli_lbl,
            "nli_score":  float(row.get("nli_entailment_score", 0) or 0),
            "similarity": float(row.get("similarity_score", 0) or 0),
            "topic_conf": float(row.get("topic_confidence", 0) or 0),
            "evidence":   str(row.get("best_evidence", "") or "").strip(),
        }
    return result


def _is_full_document(sentences_df: pd.DataFrame, esg_df: pd.DataFrame) -> bool:
    """True if sentences_df contains non-ESG sentences (i.e. the full extracted document)."""
    if "topic_label" not in sentences_df.columns:
        return False
    return (sentences_df["topic_label"] == "Non_ESG").any()


def _build_document_html(sentences_df: pd.DataFrame, esg_df: pd.DataFrame) -> str:
    """Reconstruct full document, highlighting ESG sentences."""
    if not _is_full_document(sentences_df, esg_df):
        # Only ESG sentences available (saved from old pipeline or regen without all_sentences.parquet)
        return (
            '<div style="background:#fff8e1;border-left:4px solid #ffc107;padding:12px 16px;'
            'border-radius:0 6px 6px 0;font-size:.9em;color:#555;margin-bottom:16px">'
            '⚠ Toàn văn báo cáo không khả dụng — chỉ có câu ESG được lưu. '
            'Re-run demo pipeline để xem đầy đủ (pipeline giờ lưu <code>all_sentences.parquet</code>).'
            '</div>'
            + _build_esg_only_html(esg_df)
        )

    # ESG lookup: sent_id -> (action_label, nli_label)
    esg_info: dict[str, tuple[str, str]] = {}
    for _, row in esg_df.iterrows():
        sid = str(row.get("sent_id", "") or "")
        if sid:
            action = str(row.get("action_label", "") or "")
            nli    = str(row.get("nli_label", "") or "")
            esg_info[sid] = (action, nli)

    # Sort sentences by natural document order
    df = sentences_df.copy()
    df["_sort"] = df["sent_id"].apply(_sort_key) if "sent_id" in df.columns else list(range(len(df)))
    df = df.sort_values("_sort")

    has_section = "section_title" in df.columns
    has_block   = "block_id"      in df.columns

    parts = []
    prev_section = None
    prev_block   = None

    for _, row in df.iterrows():
        sentence = str(row.get("sentence", "") or "").strip()
        if not sentence:
            continue
        sid      = str(row.get("sent_id", "") or "")
        section  = str(row.get("section_title", "") if has_section else "")
        block_id = str(row.get("block_id",      "") if has_block  else "")

        # Section heading
        if has_section and section != prev_section:
            if prev_block is not None:
                parts.append("</div>")
            prev_block = None
            if section and section not in ("UNKNOWN", "nan", "None", ""):
                parts.append(f'<div class="doc-section-title">{_h(section)}</div>')
            prev_section = section

        # Block paragraph break
        if has_block and block_id != prev_block:
            if prev_block is not None:
                parts.append("</div>")
            parts.append('<div class="doc-block">')
            prev_block = block_id
        elif not has_block and prev_block is None:
            parts.append('<div class="doc-block">')
            prev_block = "__single__"

        # Render sentence
        if sid in esg_info:
            action, nli = esg_info[sid]
            if nli == "contradiction":
                cls = "esg-sent esg-contr"
            elif action == "Implemented":
                cls = "esg-sent esg-impl"
            elif action == "Planning":
                cls = "esg-sent esg-plan"
            else:
                cls = "esg-sent esg-indet"
            parts.append(
                f'<span class="{cls}" data-id="{_h(sid)}" tabindex="0">{_h(sentence)}</span> '
            )
        else:
            parts.append(f'<span>{_h(sentence)}</span> ')

    if prev_block is not None:
        parts.append("</div>")

    return "".join(parts)


def _build_esg_only_html(esg_df: pd.DataFrame) -> str:
    """Fallback: render only ESG sentences grouped by section/block."""
    parts = []
    prev_section = None
    prev_block   = None

    df = esg_df.copy()
    df["_sort"] = df["sent_id"].apply(_sort_key) if "sent_id" in df.columns else list(range(len(df)))
    df = df.sort_values("_sort")

    for _, row in df.iterrows():
        sentence = str(row.get("sentence", "") or "").strip()
        if not sentence:
            continue
        sid     = str(row.get("sent_id", "") or "")
        section = str(row.get("section_title", "") or "")
        bid     = str(row.get("block_id", "") or "")
        action  = str(row.get("action_label", "") or "")
        nli     = str(row.get("nli_label", "") or "")

        if section != prev_section:
            if prev_block is not None:
                parts.append("</div>")
            prev_block = None
            if section and section not in ("UNKNOWN", "nan", "None", ""):
                parts.append(f'<div class="doc-section-title">{_h(section)}</div>')
            prev_section = section

        if bid != prev_block:
            if prev_block is not None:
                parts.append("</div>")
            parts.append('<div class="doc-block">')
            prev_block = bid

        if nli == "contradiction":
            cls = "esg-sent esg-contr"
        elif action == "Implemented":
            cls = "esg-sent esg-impl"
        elif action == "Planning":
            cls = "esg-sent esg-plan"
        else:
            cls = "esg-sent esg-indet"

        parts.append(
            f'<span class="{cls}" data-id="{_h(sid)}" tabindex="0">{_h(sentence)}</span> '
        )

    if prev_block is not None:
        parts.append("</div>")
    return "".join(parts)


# ── Main report generator ─────────────────────────────────────────────────────

def generate_demo_report(
    sentences_df: pd.DataFrame,
    esg_df: pd.DataFrame,
    ewri_score: EWRIScore,
    output_dir: str | Path,
    bank: str,
    year: int,
    metadata: Optional[dict] = None,
) -> Path:
    """Render an HTML report for one document. Returns the HTML path."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata        = metadata or {}
    total_sentences = len(sentences_df)
    n_esg   = len(esg_df)
    impl    = int((esg_df["action_label"] == "Implemented").sum())
    plan    = int((esg_df["action_label"] == "Planning").sum())
    indet   = int((esg_df["action_label"] == "Indeterminate").sum())
    n_evidence = int(esg_df["has_evidence"].sum()) if "has_evidence" in esg_df.columns else 0

    topic_counts = {t: int((esg_df["topic_label"] == t).sum())
                    for t in ["E", "S_labor", "S_community", "S_product", "G"]}
    entropy = calculate_topic_entropy(topic_counts)

    # ── Build HTML ────────────────────────────────────────────────────────────
    parts: list[str] = []
    parts.append(_HTML_HEAD.format(bank=_h(bank), year=year, css=_CSS))
    parts.append(f'<h1>Báo cáo phân tích ESG-Washing &mdash; {_h(bank)} {year}</h1>')

    # 1. Tóm tắt
    parts.append('<h2>1. Tóm tắt</h2>')
    stats = [
        (f"{ewri_score.ewri:.2f} / 100", "EWRI (điểm thô)"),
        (f"{total_sentences:,}",          "Tổng số câu"),
        (f"{n_esg:,} ({_fmt_pct(n_esg, total_sentences)})", "Câu ESG"),
        (_fmt_pct(n_evidence, n_esg),     "Tỷ lệ có evidence"),
        (f"{entropy:.3f}",                "Topic entropy"),
    ]
    parts.append('<div class="stat-grid">')
    for val, lbl in stats:
        parts.append(
            f'<div class="stat-card">'
            f'<div class="val">{_h(str(val))}</div>'
            f'<div class="lbl">{_h(lbl)}</div>'
            f'</div>'
        )
    parts.append('</div>')

    # 2. Phân rã EWRI
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

    # 3. Phân phối chủ đề
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

    # 4. Phân tích bằng chứng
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

    # 5. Toàn văn báo cáo
    parts.append('<h2>5. Toàn văn báo cáo (câu ESG được tô màu)</h2>')
    parts.append(
        '<p style="color:#555;font-size:.88em;margin-bottom:12px">'
        'Click vào bất kỳ câu nào được tô màu để xem chủ đề, nhãn hành động, '
        'điểm WRS và bằng chứng liên kết.</p>'
    )

    # Legend
    parts.append(
        '<div class="doc-legend">'
        '<strong>Chú thích:</strong>'
        '<span class="leg-swatch leg-impl">Implemented</span>'
        '<span class="leg-swatch leg-plan">Planning</span>'
        '<span class="leg-swatch leg-indet">Indeterminate</span>'
        '<span class="leg-swatch leg-contr">Contradiction (rủi ro cao nhất)</span>'
        '</div>'
    )

    # Full document
    parts.append('<div class="doc-body">')
    parts.append(_build_document_html(sentences_df, esg_df))
    parts.append('</div>')

    # JavaScript
    popup_data = _build_popup_data(esg_df)
    popup_json = json.dumps(popup_data, ensure_ascii=False, separators=(',', ':'))
    parts.append(f'<script>{_JS_TEMPLATE % popup_json}</script>')

    parts.append(_HTML_FOOT)

    html_path = output_dir / "report.html"
    html_path.write_text("".join(parts), encoding="utf-8")
    return html_path
