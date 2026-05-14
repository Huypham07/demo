import sys, yaml
from pathlib import Path

ROOT = Path("..") if Path("../src").exists() else Path(".")
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

FIGURES_DIR = ROOT / "thesis" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

EVIDENCE_PATH = ROOT / "outputs" / "experiments" / "evidence" / "evidence_nli.parquet"

with open(ROOT / "config" / "pipeline.yml") as f:
    CFG = yaml.safe_load(f)

from src.pipeline import ewri as ewri_mod
from src.pipeline.ewri import (
    calculate_bank_year_ewri, scores_to_dataframe, enrich_with_risk_scores,
)

ewri_mod.configure_from_dict(CFG["ewri"])

if not EVIDENCE_PATH.exists():
    raise FileNotFoundError(f"Not found: {EVIDENCE_PATH}\nRun evidence linking first.")

print(f"Loading {EVIDENCE_PATH} ...")
df = pd.read_parquet(EVIDENCE_PATH)
df = enrich_with_risk_scores(df)
ewri_scores = calculate_bank_year_ewri(df)
df_scores = scores_to_dataframe(ewri_scores).sort_values("ewri")
print(f"Loaded: {len(df):,} sentences, {len(df_scores)} bank-year observations")

BANK_TO_NH = {
    "agribank":    "NH_A",
    "bidv":        "NH_B",
    "bsc":         "NH_C",
    "mbbank":      "NH_D",
    "ocb":         "NH_E",
    "shb":         "NH_F",
    "techcombank": "NH_G",
    "vietcombank": "NH_H",
    "viettinbank": "NH_I",
    "vpbank":      "NH_J",
}
df_scores["bank_code"] = df_scores["bank"].map(BANK_TO_NH).fillna(df_scores["bank"])

# %% [markdown]
# ## Figure 1 — Spearman Correlation Heatmap

# %%
CORR_COLS = [
    "ewri", "implemented_ratio", "indeterminate_ratio",
    "planning_ratio", "evidence_ratio",
]
cc = [c for c in CORR_COLS if c in df_scores.columns]
corr = df_scores[cc].corr(method="spearman").round(2)

LABELS = {
    "ewri":                "EWRI",
    "implemented_ratio":   "Impl%",
    "indeterminate_ratio": "Indet%",
    "planning_ratio":      "Plan%",
    "evidence_ratio":      "Evidence%",
}
corr.columns = corr.index = [LABELS.get(c, c) for c in cc]

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(
    corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
    vmin=-1, vmax=1, linewidths=0.4, square=True, ax=ax,
)
ax.set_title("Correlation Matrix of EWRI Components", pad=12)
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: correlation_heatmap.png")

# %% [markdown]
# ## Figure 2 — Bank Ranking

# %%
bank_avg = (
    df_scores.groupby("bank_code")["ewri"]
    .agg(mean="mean", std="std")
    .reset_index()
    .sort_values("mean")
)
overall_mean = bank_avg["mean"].mean()

fig, ax = plt.subplots(figsize=(8, 5))
colors = [
    "#28a745" if m < 33 else "#fd7e14" if m < 37 else "#dc3545"
    for m in bank_avg["mean"]
]
ax.barh(
    bank_avg["bank_code"], bank_avg["mean"], xerr=bank_avg["std"],
    color=colors, alpha=0.85, capsize=4,
    error_kw={"linewidth": 1.2, "ecolor": "#555"},
)
ax.axvline(
    overall_mean, color="gray", linestyle="--", linewidth=1,
    label=f"Average ({overall_mean:.1f})",
)
ax.set_xlabel("Average EWRI (2020–2024)")
ax.set_title("Bank Ranking by EWRI")
ax.legend()
ax.set_xlim(25, 52)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "bank_ranking.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: bank_ranking.png")

# %% [markdown]
# ## Figure 3 — Topic Distribution

# %%
if "topic_label" not in df.columns:
    print("[Skip] topic_label column not found in evidence parquet.")
else:
    ESG_ORDER = ["G", "S_labor", "S_product", "S_community", "E"]
    TOPIC_COLORS = {
        "E": "#28a745", "S_labor": "#0f3460", "S_community": "#6610f2",
        "S_product": "#fd7e14", "G": "#e94560",
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: sentence count per topic
    tc = df["topic_label"].value_counts().reindex(ESG_ORDER, fill_value=0)
    bars = axes[0].bar(
        tc.index, tc.values,
        color=[TOPIC_COLORS[t] for t in tc.index], alpha=0.85,
    )
    for b in bars:
        axes[0].text(
            b.get_x() + b.get_width() / 2, b.get_height() + 50,
            f"{b.get_height():,}", ha="center", va="bottom", fontsize=8.5,
        )
    axes[0].set_title("ESG Topic Distribution")
    axes[0].set_ylabel("Number of sentences")
    axes[0].set_xlabel("")

    # Right: action label % per topic
    if "action_label" in df.columns:
        ta = (
            df.groupby(["topic_label", "action_label"]).size()
            .unstack(fill_value=0)
            .reindex(ESG_ORDER, fill_value=0)
        )
        ta_pct = ta.div(ta.sum(axis=1), axis=0) * 100
        ta_pct[["Implemented", "Planning", "Indeterminate"]].plot(
            kind="bar", ax=axes[1], rot=20,
            color=["#28a745", "#fd7e14", "#dc3545"], alpha=0.85,
        )
        axes[1].set_title("Action Label Distribution by Topic (%)")
        axes[1].set_ylabel("%")
        axes[1].set_ylim(0, 100)
        axes[1].legend(title="Action", bbox_to_anchor=(1.02, 1), loc="upper left")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "topic_distribution.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: topic_distribution.png")
