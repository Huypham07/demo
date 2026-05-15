import sys, yaml
from pathlib import Path

ROOT = Path("..") if Path("../src").exists() else Path(".")
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import torch
from sklearn.metrics import classification_report
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

FIGURES_DIR = ROOT / "thesis" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

with open(ROOT / "config" / "pipeline.yml") as f:
    CFG = yaml.safe_load(f)

TOPIC_LABELS  = CFG["model"]["topic"]["labels"]
ACTION_LABELS = CFG["model"]["actionability"]["labels"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"ROOT       : {ROOT.resolve()}")
print(f"FIGURES_DIR: {FIGURES_DIR.resolve()}")
print(f"Device     : {DEVICE}")

# %% [markdown]
# ## 1. RQ1 — Phân loại: Baseline vs Neuro-Symbolic

# %%
# ── Hàm tiện ích ───────────────────────────────────────────────────────────────

def run_inference(model, tokenizer, sentences, batch_size=64):
    """Trả về (predicted_labels, confidence_scores)."""
    model.to(DEVICE).eval()
    id2label = model.config.id2label
    all_preds, all_confs = [], []
    for i in tqdm(range(0, len(sentences), batch_size), desc="Inference", leave=False):
        batch = sentences[i : i + batch_size]
        enc = tokenizer(batch, return_tensors="pt", truncation=True,
                        padding=True, max_length=128)
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        with torch.no_grad():
            probs = torch.softmax(model(**enc).logits, dim=-1)
        preds = torch.argmax(probs, dim=-1).cpu().tolist()
        confs = probs.max(dim=-1).values.cpu().tolist()
        all_preds.extend([id2label[p] for p in preds])
        all_confs.extend(confs)
    return all_preds, all_confs


def evaluate(model_id_or_path, test_df, text_col, label_col, task_labels, tag):
    """Load model, run inference, print report, return dict report."""
    print(f"\n{'─'*55}\n{tag}: {model_id_or_path}\n{'─'*55}")
    tok   = AutoTokenizer.from_pretrained(model_id_or_path, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(model_id_or_path)
    preds, _ = run_inference(model, tok, test_df[text_col].tolist())
    print(classification_report(test_df[label_col], preds, labels=task_labels, digits=3))
    return classification_report(test_df[label_col], preds,
                                 labels=task_labels, output_dict=True)


def _resolve(local_path, hf_id):
    p = ROOT / local_path
    return str(p) if p.exists() else hf_id

# %%
# ── 1-A  Chủ đề ESG ───────────────────────────────────────────────────────────

topic_test = pd.read_parquet(ROOT / "data/labels/topic/test.parquet")
print(f"Topic test: {len(topic_test):,} samples\n{topic_test['label'].value_counts()}")

TOPIC_NS   = _resolve("outputs/models/topic_classifier/final",
                       CFG["model"]["topic"]["hf_model_id"])
TOPIC_BASE = _resolve("outputs/models/topic_baseline/final", None)

report_topic_ns = evaluate(TOPIC_NS, topic_test, "sentence", "label",
                            TOPIC_LABELS, "Topic – Neuro-Symbolic")

if TOPIC_BASE:
    report_topic_base = evaluate(TOPIC_BASE, topic_test, "sentence", "label",
                                  TOPIC_LABELS, "Topic – Baseline")
else:
    print("\n⚠  Baseline topic model không tìm thấy — dùng số liệu từ luận văn (Bảng 4.3).")
    report_topic_base = {
        "E":           {"f1-score": 0.73}, "S_labor":     {"f1-score": 0.78},
        "S_community": {"f1-score": 0.73}, "S_product":   {"f1-score": 0.66},
        "G":           {"f1-score": 0.79}, "Non_ESG":     {"f1-score": 0.89},
        "macro avg":   {"f1-score": 0.76}, "accuracy": 0.870,
    }

# %%
# ── 1-B  Mức độ hành động ─────────────────────────────────────────────────────

action_test = pd.read_parquet(ROOT / "data/labels/action/test.parquet")
print(f"Action test: {len(action_test):,} samples\n{action_test['label'].value_counts()}")

ACTION_NS   = _resolve("outputs/models/action_classifier/final",
                        CFG["model"]["actionability"]["hf_model_id"])
ACTION_BASE = _resolve("outputs/models/action_baseline/final", None)

report_action_ns = evaluate(ACTION_NS, action_test, "sentence", "label",
                             ACTION_LABELS, "Action – Neuro-Symbolic")

if ACTION_BASE:
    report_action_base = evaluate(ACTION_BASE, action_test, "sentence", "label",
                                   ACTION_LABELS, "Action – Baseline")
else:
    print("\n⚠  Baseline action model không tìm thấy — dùng số liệu từ luận văn (Bảng 4.4).")
    report_action_base = {
        "Implemented":   {"f1-score": 0.80}, "Planning":      {"f1-score": 0.63},
        "Indeterminate": {"f1-score": 0.88},
        "macro avg":     {"f1-score": 0.77}, "accuracy": 0.862,
    }

# %%
# ── 1-C  Hình so sánh F1 (figures/rq1_perclass_*.png) ─────────────────────────

def plot_f1_comparison(labels, base_rep, ns_rep, title, fname):
    base_f1 = [base_rep.get(l, {}).get("f1-score", 0) for l in labels]
    ns_f1   = [ns_rep.get(l,   {}).get("f1-score", 0) for l in labels]
    x, w = np.arange(len(labels)), 0.35
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - w/2, base_f1, w, label="Baseline",       color="#6c757d", alpha=0.85)
    ax.bar(x + w/2, ns_f1,   w, label="Neuro-Symbolic", color="#0f3460", alpha=0.90)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("F1-score"); ax.set_ylim(0, 1.08)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.set_title(title); ax.legend(loc="lower right")
    ax.axhline(0.8, color="#e94560", linestyle="--", linewidth=0.8, alpha=0.6)
    for xi, (b, n) in enumerate(zip(base_f1, ns_f1)):
        ax.text(xi - w/2, b + 0.01, f"{b:.2f}", ha="center", va="bottom", fontsize=7.5)
        ax.text(xi + w/2, n + 0.01, f"{n:.2f}", ha="center", va="bottom", fontsize=7.5)
    plt.tight_layout()
    for ext in ("png", "eps"):
        plt.savefig(FIGURES_DIR / f"{fname}.{ext}", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {fname}.png / .eps")

plot_f1_comparison(TOPIC_LABELS,  report_topic_base,  report_topic_ns,
                   "F1-score phân loại chủ đề ESG: Baseline vs Neuro-Symbolic",
                   "rq1_perclass_topic")

plot_f1_comparison(ACTION_LABELS, report_action_base, report_action_ns,
                   "F1-score phân loại mức độ hành động: Baseline vs Neuro-Symbolic",
                   "rq1_perclass_action")

# %% [markdown]
# ## 2. RQ2 — So sánh phương pháp liên kết bằng chứng

# %%
from src.pipeline.evidence_extract import evidence_extract

CORPUS_PATH = ROOT / "data/corpus/actionability_sentences.parquet"
if not CORPUS_PATH.exists():
    raise FileNotFoundError(
        f"{CORPUS_PATH} không tồn tại.\n"
        "Chạy phần Phụ lục ở cuối notebook trước."
    )

df_corpus = pd.read_parquet(CORPUS_PATH)
print(f"Corpus: {len(df_corpus):,} ESG sentences")

OUT_EV = ROOT / "outputs/experiments/evidence"
OUT_EV.mkdir(parents=True, exist_ok=True)

# %%
rq2 = {}
for variant in ["nli", "window", "no_nli"]:
    cache = OUT_EV / f"evidence_{variant}.parquet"
    if cache.exists():
        df_v = pd.read_parquet(cache)
        print(f"[{variant}] loaded from cache")
    else:
        print(f"[{variant}] computing…")
        df_v = evidence_extract(df_corpus.copy(), variant=variant, config=CFG)
        df_v.to_parquet(cache, index=False)

    n_total = len(df_v)
    n_ev    = int(df_v["has_evidence"].sum()) if "has_evidence" in df_v.columns else 0
    avg_sim = float(df_v["similarity_score"].mean()) if "similarity_score" in df_v.columns else 0.0
    nli_cnt = df_v["nli_label"].value_counts(dropna=False).to_dict() \
              if "nli_label" in df_v.columns else {}
    rq2[variant] = dict(
        evidence_rate_pct = round(n_ev / n_total * 100, 1),
        avg_similarity    = round(avg_sim, 4),
        entailment_pct    = round(nli_cnt.get("entailment", 0)    / max(n_ev, 1) * 100, 1),
        contradiction_pct = round(nli_cnt.get("contradiction", 0) / max(n_ev, 1) * 100, 1),
    )

print("\n=== RQ2 Summary ===")
print(pd.DataFrame(rq2).T.to_string())

# %% [markdown]
# ## 2b. Grid Search tham số EWRI
#
# Chạy grid search để tìm bộ tham số (P_action, λ, C) tối đa hoá CV = Std/Mean của EWRI.
# Ràng buộc: P(Impl) < P(Plan) < P(Indet),  λ(Impl) > λ(Plan) > λ(Indet),  P(Indet)×C ≤ 1.0
#
# **Kết quả lưu vào** `outputs/ewri_grid_search_results.csv`.
# **Sau khi chạy**, cập nhật tham số tối ưu vào `config/pipeline.yml`.

# %%
from src.pipeline.ewri_grid_search import run_grid_search
from src.pipeline import ewri as ewri_mod

_gs_input = OUT_EV / "evidence_nli.parquet"
_gs_out   = ROOT / "outputs" / "ewri_grid_search_results.csv"

if _gs_input.exists():
    df_gs = pd.read_parquet(_gs_input)
    gs_results, gs_best = run_grid_search(df_gs)

    gs_results.to_csv(_gs_out, index=False)
    print(f"\n{'='*55}")
    print(f"Best CV   : {gs_best['CV']:.4f}")
    print(f"Std EWRI  : {gs_best['Std']:.4f}")
    print(f"Mean EWRI : {gs_best['Mean']:.2f}")
    print(f"\nOptimal parameters (copy vào config/pipeline.yml):")
    print(f"  action_penalty     : {gs_best['P']}")
    print(f"  evidence_sensitivity: {gs_best['Lambda']}")
    print(f"  contradiction_amplifier: {gs_best['C']}")
    print(f"\nSaved: {_gs_out}")

    # Áp dụng tham số tối ưu ngay trong notebook cho RQ3
    ewri_mod.configure_ewri({
        "action_penalty":          gs_best["P"],
        "evidence_sensitivity":    gs_best["Lambda"],
        "contradiction_amplifier": gs_best["C"],
    })
    print("ewri module patched with optimal parameters.")
else:
    print(f"[Skip] {_gs_input} not found — chạy bước 3a trước.")

# %% [markdown]
# ## 3. RQ3 — EWRI: 50 quan sát ngân hàng-năm

# %%
from src.pipeline.ewri import (
    calculate_bank_year_ewri, scores_to_dataframe,
    enrich_with_risk_scores, print_ewri_summary,
)

df_nli = pd.read_parquet(OUT_EV / "evidence_nli.parquet")
df_nli = enrich_with_risk_scores(df_nli)

ewri_scores = calculate_bank_year_ewri(df_nli)
df_scores   = scores_to_dataframe(ewri_scores).sort_values("ewri")
print_ewri_summary(df_scores)

# %%
# ── Bảng phân rã đóng góp (Bảng 4.5) ─────────────────────────────────────────
print(f"\nEWRI trung bình: {df_scores['ewri'].mean():.2f}  std={df_scores['ewri'].std():.2f}")
print(f"Phạm vi: [{df_scores['ewri'].min():.2f}, {df_scores['ewri'].max():.2f}]\n")
mu = df_scores["ewri"].mean()
for col, label in [("contrib_indeterminate","Indeterminate"),
                   ("contrib_implemented",  "Implemented"),
                   ("contrib_planning",     "Planning")]:
    if col in df_scores.columns:
        m = df_scores[col].mean()
        print(f"  {label:15s}  {m:.2f} pts  ({m/mu*100:.1f}%  EWRI)  "
              f"std={df_scores[col].std():.2f}")

# %%
# ── Hình: Correlation heatmap (Hình 4.3) ──────────────────────────────────────
CORR_COLS = ["ewri", "implemented_ratio", "indeterminate_ratio", "planning_ratio",
             "evidence_ratio", "avg_evidence_strength", "topic_entropy"]
cc = [c for c in CORR_COLS if c in df_scores.columns]
corr = df_scores[cc].corr(method="spearman").round(2)

LABELS_VI = {
    "ewri": "EWRI", "implemented_ratio": "Impl%",
    "indeterminate_ratio": "Indet%", "planning_ratio": "Plan%",
    "evidence_ratio": "Ev rate", "avg_evidence_strength": "Ev strength",
    "topic_entropy": "Topic H",
}
corr.columns = corr.index = [LABELS_VI.get(c, c) for c in cc]

fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
            vmin=-1, vmax=1, linewidths=0.4, square=True, ax=ax)
ax.set_title("Spearman correlation — EWRI components", pad=12)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: correlation_heatmap.png")

# %%
# ── Hình: Bank ranking (Hình 4.4) ─────────────────────────────────────────────
bank_avg = (df_scores.groupby("bank")["ewri"]
            .agg(mean="mean", std="std")
            .reset_index().sort_values("mean"))

fig, ax = plt.subplots(figsize=(8, 5))
colors = ["#28a745" if m < 43 else "#fd7e14" if m < 47 else "#dc3545"
          for m in bank_avg["mean"]]
ax.barh(bank_avg["bank"], bank_avg["mean"], xerr=bank_avg["std"],
        color=colors, alpha=0.85, capsize=4,
        error_kw={"linewidth": 1.2, "ecolor": "#555"})
ax.axvline(bank_avg["mean"].mean(), color="gray", linestyle="--",
           linewidth=1, label=f"Trung bình ({bank_avg['mean'].mean():.1f})")
ax.set_xlabel("EWRI trung bình (2020–2024)")
ax.set_title("Xếp hạng ngân hàng theo EWRI  (thấp = minh bạch ESG hơn)")
ax.legend(); ax.set_xlim(35, 60)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "bank_ranking.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: bank_ranking.png")

# %%
# ── Hình: Topic distribution (Hình 4.5) ───────────────────────────────────────
topic_col  = "topic_label"  if "topic_label"  in df_nli.columns else None
action_col = "action_label" if "action_label" in df_nli.columns else None

if topic_col:
    ESG_ORDER = ["G", "S_labor", "S_product", "S_community", "E"]
    TOPIC_COLORS = {"E": "#28a745", "S_labor": "#0f3460", "S_community": "#6610f2",
                    "S_product": "#fd7e14", "G": "#e94560"}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Trái: số lượng câu theo chủ đề
    tc = df_nli[topic_col].value_counts().reindex(ESG_ORDER, fill_value=0)
    bars = axes[0].bar(tc.index, tc.values,
                       color=[TOPIC_COLORS[t] for t in tc.index], alpha=0.85)
    for b in bars:
        axes[0].text(b.get_x() + b.get_width()/2, b.get_height() + 50,
                     f"{b.get_height():,}", ha="center", va="bottom", fontsize=8.5)
    axes[0].set_title("Phân phối chủ đề ESG (corpus)")
    axes[0].set_ylabel("Số câu"); axes[0].set_xlabel("")

    # Phải: % nhãn hành động theo chủ đề
    if action_col:
        ta = (df_nli.groupby([topic_col, action_col]).size()
              .unstack(fill_value=0).reindex(ESG_ORDER, fill_value=0))
        ta_pct = ta.div(ta.sum(axis=1), axis=0) * 100
        ta_pct[["Implemented","Planning","Indeterminate"]].plot(
            kind="bar", ax=axes[1], rot=20,
            color=["#28a745", "#fd7e14", "#dc3545"], alpha=0.85
        )
        axes[1].set_title("Phân bố nhãn hành động theo chủ đề (%)")
        axes[1].set_ylabel("%"); axes[1].set_ylim(0, 100)
        axes[1].legend(title="Action", bbox_to_anchor=(1.02, 1), loc="upper left")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "topic_distribution.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: topic_distribution.png")

# %% [markdown]
# ---
# ## Phụ lục: Chạy corpus pipeline (nếu chưa có actionability_sentences.parquet)
#
# Chạy phần này **một lần** trước khi làm RQ2/RQ3.
# Yêu cầu raw OCR text trong `data/extracted/raw_ocr_annual_report/` (giải nén từ `.zip`).

# %%
# import zipfile
# with zipfile.ZipFile(ROOT / "data/extracted/raw_ocr_annual_report.zip") as z:
#     z.extractall(ROOT / "data/extracted")

# %%
# from src.pipeline.pipeline import ESGWashingPipeline
# import pandas as pd
#
# pipeline = ESGWashingPipeline(config_path=str(ROOT / "config/pipeline.yml"))
#
# # 1. Build corpus
# pipeline.build_corpus(raw_txt_path=str(ROOT / "data/extracted/raw_ocr_annual_report"))
#
# # 2. Topic + Action classification
# sentences_df = pd.read_parquet(ROOT / "data/corpus/sentences.parquet")
# topic_df     = pipeline.topic_classification(sentences_df)
# action_df    = pipeline.actionability_classification(topic_df)
#
# out = ROOT / "data/corpus/actionability_sentences.parquet"
# action_df.to_parquet(out, index=False)
# print(f"Saved {len(action_df):,} ESG sentences → {out}")
