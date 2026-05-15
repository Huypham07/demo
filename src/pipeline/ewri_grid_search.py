import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

P_ACTION_GRIDS = {
    "Implemented": [0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20],
    "Planning": [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55],
    "Indeterminate": [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90],
}

LAMBDA_GRIDS = {
    "Implemented": [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00],
    "Planning": [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85],
    "Indeterminate": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50],
}

C_GRIDS = [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8]

_ACTION_KEYS = ["Implemented", "Planning", "Indeterminate"]


def _build_valid_combos(grids: dict, ordering_check) -> list[dict]:
    keys, values = zip(*grids.items())
    combos = [dict(zip(keys, v)) for v in itertools.product(*values)]
    return [c for c in combos if ordering_check(c)]


def _kendall_w_batch(ewri_matrix: np.ndarray) -> np.ndarray:
    """
    Compute Kendall's W (coefficient of concordance) for each parameter combo.

    ewri_matrix : (n_combos, n_banks, n_years)
        EWRI score for each combo, bank, and year.

    Returns W : (n_combos,) in [0, 1].
        W = 1 → perfect agreement in bank rankings across years.
        W = 0 → no concordance.

    Objective rationale: if EWRI captures stable institutional ESG-washing
    characteristics, bank rankings should be consistent across reporting years.
    This criterion has a genuine interior optimum — parameters too small collapse
    all scores toward zero (rankings become noisy); parameters too large amplify
    random variation rather than systematic bank differences.
    """
    n_combos, n_banks, n_years = ewri_matrix.shape
    if n_banks < 2 or n_years < 2:
        return np.zeros(n_combos, dtype=np.float32)

    # Rank banks within each year (axis=1 = bank axis). Double argsort → ordinal rank.
    ranks = np.argsort(np.argsort(ewri_matrix, axis=1), axis=1).astype(np.float64) + 1

    # R[combo, bank] = sum of ranks across all years
    R = ranks.sum(axis=2)                          # (n_combos, n_banks)
    R_mean = n_years * (n_banks + 1) / 2
    S = ((R - R_mean) ** 2).sum(axis=1)            # (n_combos,)
    W = 12.0 * S / (n_years ** 2 * (n_banks ** 3 - n_banks))
    return W.astype(np.float32)


def run_grid_search(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    actions = df["action_label"].astype(str).to_numpy()
    nli = (df["nli_label"].astype(str).to_numpy()
           if "nli_label" in df.columns else np.full(len(df), "", dtype=object))
    is_contr = (nli == "contradiction")
    es = np.where(nli == "entailment", 1.0, 0.0).astype(np.float32)

    valid_p = _build_valid_combos(
        P_ACTION_GRIDS,
        lambda c: c["Implemented"] < c["Planning"] < c["Indeterminate"],
    )
    valid_l = _build_valid_combos(
        LAMBDA_GRIDS,
        lambda c: c["Implemented"] > c["Planning"] > c["Indeterminate"],
    )
    n_p, n_l = len(valid_p), len(valid_l)
    print(f"P combos: {n_p}  |  L combos: {n_l}  |  C values: {len(C_GRIDS)}")
    print(f"Total combinations: {n_p * n_l * len(C_GRIDS):,}  (WRS-bound filter: P_indet × C ≤ 1)")

    mask = np.column_stack([(actions == k).astype(np.float32) for k in _ACTION_KEYS])
    P_vals = np.array([[p[k] for k in _ACTION_KEYS] for p in valid_p], dtype=np.float32)
    P_mat = P_vals @ mask.T   # (n_p, n_sentences)
    L_vals = np.array([[l[k] for k in _ACTION_KEYS] for l in valid_l], dtype=np.float32)
    L_mat = L_vals @ mask.T   # (n_l, n_sentences)

    # Structured (bank × year) aggregation for Kendall's W
    banks = sorted(df["bank"].unique())
    years = sorted(df["year"].unique())
    n_banks, n_years = len(banks), len(years)
    bank_to_idx = {b: i for i, b in enumerate(banks)}
    year_to_idx = {y: i for i, y in enumerate(years)}
    group_idx = np.array([bank_to_idx[b] * n_years + year_to_idx[y]
                          for b, y in zip(df["bank"], df["year"])])
    n_groups_by = n_banks * n_years
    one_hot_by = np.zeros((len(df), n_groups_by), dtype=np.float32)
    one_hot_by[np.arange(len(df)), group_idx] = 1.0
    counts_by = np.maximum(one_hot_by.sum(axis=0), 1.0)

    # Flat bank-year aggregation for mean/std/CV reporting
    bank_year = df[["bank", "year"]].astype(str).agg("_".join, axis=1).to_numpy()
    unique_keys, inverse = np.unique(bank_year, return_inverse=True)
    n_groups = len(unique_keys)
    one_hot = np.zeros((len(df), n_groups), dtype=np.float32)
    one_hot[np.arange(len(df)), inverse] = 1.0
    counts = np.maximum(one_hot.sum(axis=0), 1.0)

    best_w = -np.inf
    best_params: dict = {}
    rows: list[dict] = []

    for p_idx in tqdm(range(n_p), desc="P_action"):
        p_dict = valid_p[p_idx]
        p_indet = p_dict["Indeterminate"]
        p_arr = P_mat[p_idx]

        base_all = p_arr[None, :] * (1.0 - L_mat * es[None, :])  # (n_l, n_sentences)

        for c_val in C_GRIDS:
            if p_indet * c_val > 1.0:   # WRS upper-bound constraint
                continue

            wrs_all = np.where(
                is_contr[None, :],
                np.minimum(base_all * c_val, 1.0),
                base_all,
            ).clip(0.0, 1.0)

            # ── Primary objective: Kendall's W ──────────────────────────────
            ewri_by = (wrs_all @ one_hot_by / counts_by[None, :]) * 100.0  # (n_l, n_banks*n_years)
            ewri_mat = ewri_by.reshape(n_l, n_banks, n_years)               # (n_l, n_banks, n_years)
            w_all = _kendall_w_batch(ewri_mat)                              # (n_l,)

            # ── Secondary metrics (CV, mean, std) ───────────────────────────
            ewri_cv = (wrs_all @ one_hot / counts[None, :]) * 100.0
            std_all = ewri_cv.std(axis=1)
            mean_all = ewri_cv.mean(axis=1)
            cv_all = np.where(mean_all > 0, std_all / mean_all, 0.0)

            best_l = int(np.argmax(w_all))
            if w_all[best_l] > best_w:
                best_w = float(w_all[best_l])
                best_params = {
                    "P": p_dict,
                    "Lambda": valid_l[best_l],
                    "C": c_val,
                    "Mean": float(mean_all[best_l]),
                    "Std": float(std_all[best_l]),
                    "CV": float(cv_all[best_l]),
                    "KendallW": best_w,
                }

            for l_idx in range(n_l):
                l_dict = valid_l[l_idx]
                rows.append({
                    "P_Implemented": p_dict["Implemented"],
                    "P_Planning": p_dict["Planning"],
                    "P_Indeterminate": p_dict["Indeterminate"],
                    "L_Implemented": l_dict["Implemented"],
                    "L_Planning": l_dict["Planning"],
                    "L_Indeterminate": l_dict["Indeterminate"],
                    "C_amplifier": c_val,
                    "Mean_EWRI": float(mean_all[l_idx]),
                    "Std_EWRI": float(std_all[l_idx]),
                    "CV_EWRI": float(cv_all[l_idx]),
                    "KendallW_EWRI": float(w_all[l_idx]),
                })

    results_df = (pd.DataFrame(rows)
                  .sort_values("KendallW_EWRI", ascending=False)
                  .reset_index(drop=True))
    return results_df, best_params


def main():
    parser = argparse.ArgumentParser(
        description="EWRI grid search — objective: Kendall's W (rank concordance across years)."
    )
    parser.add_argument("--input", default="outputs/experiments/evidence/evidence_nli.parquet")
    parser.add_argument("--output", default="outputs/ewri_grid_search_results.csv")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[Error] Input not found: {input_path}")
        return

    print(f"Loading {input_path} ...")
    df = pd.read_parquet(input_path)
    print(f"Rows: {len(df):,}  |  Banks: {df['bank'].nunique()}  |  Years: {df['year'].nunique()}")

    results_df, best = run_grid_search(df)

    print("\n" + "=" * 60)
    print("EWRI GRID SEARCH  (objective = Kendall's W)")
    print("=" * 60)
    print(f"Best Kendall's W : {best['KendallW']:.4f}")
    print(f"Mean EWRI        : {best['Mean']:.2f}")
    print(f"Std EWRI         : {best['Std']:.4f}")
    print(f"CV (secondary)   : {best['CV']:.4f}")
    print(f"\nOptimal parameters:")
    print(f"  P_action : {best['P']}")
    print(f"  Lambda   : {best['Lambda']}")
    print(f"  C        : {best['C']}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out, index=False)
    print(f"\nSaved: {out}  ({len(results_df):,} rows)")


if __name__ == "__main__":
    main()
