import pandas as pd
import sys
sys.path.insert(0, ".")
from src.pipeline.ewri import enrich_with_risk_scores, calculate_bank_year_ewri, scores_to_dataframe
from scipy import stats

df = pd.read_parquet("outputs/experiments/evidence/evidence_nli.parquet")
df = enrich_with_risk_scores(df)
scores = calculate_bank_year_ewri(df)
df_s = scores_to_dataframe(scores)

ewri = df_s["ewri"]
print(f"EWRI: mean={ewri.mean():.3f}, std={ewri.std():.3f}")
print(f"Range: [{ewri.min():.2f}, {ewri.max():.2f}]")

me = ewri.mean()
ci = df_s["contrib_indeterminate"].mean(); si = df_s["contrib_indeterminate"].std()
cp = df_s["contrib_planning"].mean();      sp = df_s["contrib_planning"].std()
cm = df_s["contrib_implemented"].mean();   sm = df_s["contrib_implemented"].std()
print(f"\nDecomposition:")
print(f"  Indet: {ci:.2f} std={si:.2f} = {ci/me*100:.1f}%")
print(f"  Impl:  {cm:.2f} std={sm:.2f} = {cm/me*100:.1f}%")
print(f"  Plan:  {cp:.2f} std={sp:.2f} = {cp/me*100:.1f}%")

r_indet, _ = stats.pearsonr(df_s["indeterminate_ratio"], ewri)
r_impl,  _ = stats.pearsonr(df_s["implemented_ratio"],   ewri)
r_ev,    _ = stats.pearsonr(df_s["evidence_ratio"],       ewri)
r_plan,  _ = stats.pearsonr(df_s["planning_ratio"],       ewri)
print(f"\nCorrelations:")
print(f"  r(Indet, EWRI) = {r_indet:.3f}")
print(f"  r(Impl,  EWRI) = {r_impl:.3f}")
print(f"  r(Ev,    EWRI) = {r_ev:.3f}")
print(f"  r(Plan,  EWRI) = {r_plan:.3f}")

BANK_MAP = {
    "agribank": "NH_A", "bidv": "NH_B", "bsc": "NH_C", "mbbank": "NH_D",
    "ocb": "NH_E", "shb": "NH_F", "techcombank": "NH_G",
    "vietcombank": "NH_H", "viettinbank": "NH_I", "vpbank": "NH_J",
}
bank_agg = df_s.groupby("bank").agg(
    total=("total_sentences", "sum"),
    ewri_mean=("ewri", "mean"), ewri_std=("ewri", "std"),
    impl=("implemented_ratio", "mean"),
    indet=("indeterminate_ratio", "mean"),
    ev=("evidence_ratio", "mean"),
).round(3).sort_values("ewri_mean")

print("\nBank ranking:")
for i, (bank, row) in enumerate(bank_agg.iterrows(), 1):
    nh = BANK_MAP.get(bank, bank)
    total = int(row["total"])
    print(f"  {i} {nh} {total:,} EWRI={row['ewri_mean']:.2f} Std={row['ewri_std']:.2f} "
          f"Impl={row['impl']*100:.1f}% Indet={row['indet']*100:.1f}% Ev={row['ev']*100:.1f}%")
