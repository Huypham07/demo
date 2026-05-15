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


def run_grid_search(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    # input
    actions = df["action_label"].astype(str).to_numpy()
    has_ev = (df["has_evidence"].astype(bool).to_numpy()
              if "has_evidence" in df.columns else np.zeros(len(df), dtype=bool))
    nli = (df["nli_label"].astype(str).to_numpy()
           if "nli_label" in df.columns else np.full(len(df), "", dtype=object))
    is_contr = (nli == "contradiction")

    # es = 1 only when NLI confirms entailment (genuine evidence support)
    es = np.where(nli == "entailment", 1.0, 0.0).astype(np.float32)

    # valid parameter combinations
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
    print(f"Max combinations: {n_p * n_l * len(C_GRIDS):,}  (WRS-bound filter applied per C)")

    mask = np.column_stack([
        (actions == k).astype(np.float32) for k in _ACTION_KEYS
    ])

    # P_mat[i, :] = p_arr for i P combo  ->  shape (n_p, n_sentences)
    P_vals = np.array([[p[k] for k in _ACTION_KEYS] for p in valid_p], dtype=np.float32)
    P_mat = P_vals @ mask.T # (n_p, n_sentences)

    # L_mat[j, :] = L_arr for j L combo  ->  shape (n_l, n_sentences)
    L_vals = np.array([[l[k] for k in _ACTION_KEYS] for l in valid_l], dtype=np.float32)
    L_mat = L_vals @ mask.T # (n_l, n_sentences)

    bank_year = df[["bank", "year"]].astype(str).agg("_".join, axis=1).to_numpy()
    unique_keys, inverse = np.unique(bank_year, return_inverse=True)
    n_groups = len(unique_keys)

    one_hot = np.zeros((len(df), n_groups), dtype=np.float32)
    one_hot[np.arange(len(df)), inverse] = 1.0
    counts = np.maximum(one_hot.sum(axis=0), 1.0)  # (n_groups,)

    # grid search
    best_cv = -np.inf
    best_params: dict = {}
    rows: list[dict] = []

    for p_idx in tqdm(range(n_p), desc="P_action"):
        p_dict = valid_p[p_idx]
        p_indet = p_dict["Indeterminate"]
        p_arr = P_mat[p_idx]   # (n_sentences,)

        # base_wrs for all L combos at once: (n_l, n_sentences)
        # base_wrs[j, i] = p_arr[i] x (1 − L_arr[j, i] x es[i])
        base_all = p_arr[None, :] * (1.0 - L_mat * es[None, :])

        for c_val in C_GRIDS:
            if p_indet * c_val > 1.0:   # WRS-bound constraint
                continue

            # Apply contradiction amplifier: (n_l, n_sentences)
            wrs_all = np.where(
                is_contr[None, :],
                np.minimum(base_all * c_val, 1.0),
                base_all,
            ).clip(0.0, 1.0)

            # Aggregate -> (n_l, n_groups) -> EWRI x 100
            ewri_all = (wrs_all @ one_hot / counts[None, :]) * 100.0

            std_all = ewri_all.std(axis=1) # (n_l,)
            mean_all = ewri_all.mean(axis=1) # (n_l,)
            cv_all = np.where(mean_all > 0, std_all / mean_all, 0.0)

            # Track best
            best_l = int(np.argmax(cv_all))
            if cv_all[best_l] > best_cv:
                best_cv = cv_all[best_l]
                best_params = {
                    "P": p_dict,
                    "Lambda": valid_l[best_l],
                    "C": c_val,
                    "Mean": float(mean_all[best_l]),
                    "Std": float(std_all[best_l]),
                    "CV": float(best_cv),
                }

            # Collect all L results for this (P, C)
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
                })

    results_df = pd.DataFrame(rows).sort_values("CV_EWRI", ascending=False).reset_index(drop=True)
    return results_df, best_params


def main():
    parser = argparse.ArgumentParser(description="Vectorised EWRI grid search.")
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

    print("\n" + "=" * 55)
    print("EWRI GRID SEARCH  (objective = CV = Std / Mean)")
    print("=" * 55)
    print(f"Best CV   : {best['CV']:.4f}")
    print(f"Std EWRI  : {best['Std']:.4f}")
    print(f"Mean EWRI : {best['Mean']:.2f}")
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
