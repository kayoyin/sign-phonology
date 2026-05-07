"""Per-pair t-test of model phonological sensitivity.

For each minimal pair (s1, s2) the script builds two distributions of cosine
similarities over a model's feature vectors (extracted by ``i3d/extract_features.py``
or ``stgcn/extract_features.py``):

  - ``D_same``  : sim(<s1 by signer A>, <s1 by signer B>)
  - ``D_diff``  : sim(<s1 by signer A>, <s2 by signer B>)

and runs an independent-samples t-test. We expect ``D_same > D_diff`` for a
phonologically sensitive model.

A control loop also samples ``--n_random_pairs`` random non-minimal pairs from
the same gloss pool, to verify the effect is specific to minimal pairs and not
a generic property of any sign pair.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity


def cosine(vec1, vec2):
    return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))


def mean_pairwise_cosine_similarity(vectors):
    """Mean upper-triangle cosine similarity across a list of vectors."""
    matrix = np.array(vectors)
    sim = cosine_similarity(matrix)
    upper = sim[np.triu_indices(len(matrix), k=1)]
    return float(np.mean(upper)) if upper.size else np.nan


def pair_ttest(row, feature_db, max_pairs=5):
    """Independent t-test on intra- vs. inter-sign cosine similarities."""
    gloss1 = row["ASLCitizenGLOSS1"].upper().strip()
    gloss2 = row["ASLCitizenGLOSS2"].upper().strip()
    users1 = feature_db.loc[feature_db["gloss"] == gloss1, "user"].unique()
    users2 = feature_db.loc[feature_db["gloss"] == gloss2, "user"].unique()
    users = list(set(users1) & set(users2))
    if len(users) < 5:
        return np.nan, np.nan, np.nan, np.nan

    cos_inter = []
    cos_intra = []
    for userA in users:
        if len(cos_inter) >= max_pairs:
            break
        for userB in users:
            if userA == userB:
                continue
            v_a = feature_db.loc[
                (feature_db["gloss"] == gloss1) & (feature_db["user"] == userA), "features"
            ].values[0]
            v_b1 = feature_db.loc[
                (feature_db["gloss"] == gloss2) & (feature_db["user"] == userB), "features"
            ].values[0]
            v_b2 = feature_db.loc[
                (feature_db["gloss"] == gloss1) & (feature_db["user"] == userB), "features"
            ].values[0]
            sim_inter = cosine(v_a, v_b1)
            sim_intra = cosine(v_a, v_b2)
            if np.isnan(sim_inter) or np.isnan(sim_intra):
                continue
            cos_inter.append(sim_inter)
            cos_intra.append(sim_intra)
            if len(cos_inter) >= max_pairs:
                break

    if not cos_inter:
        return np.nan, np.nan, np.nan, np.nan
    res = stats.ttest_ind(np.array(cos_inter), np.array(cos_intra))
    return float(np.mean(cos_inter)), float(np.mean(cos_intra)), float(res.statistic), float(res.pvalue)


def build_ttest_table(pairs_df, feature_db):
    out = pd.DataFrame()
    out["ASLCitizenGLOSS1"] = pairs_df["ASLCitizenGLOSS1"]
    out["ASLCitizenGLOSS2"] = pairs_df["ASLCitizenGLOSS2"]
    out["cos_sim_inter"], out["cos_sim_intra"], out["t_stat"], out["p_value"] = zip(
        *pairs_df.apply(lambda r: pair_ttest(r, feature_db), axis=1)
    )
    return out


def sample_random_nonminimal_pairs(minimal_pairs, n_pairs, seed=42):
    """Sample ``n_pairs`` random gloss pairs that are NOT minimal pairs."""
    pool = pd.unique(
        pd.concat([
            minimal_pairs["ASLCitizenGLOSS1"].astype(str).str.upper().str.strip(),
            minimal_pairs["ASLCitizenGLOSS2"].astype(str).str.upper().str.strip(),
        ])
    )
    forbidden = set()
    for _, r in minimal_pairs.iterrows():
        a = str(r["ASLCitizenGLOSS1"]).upper().strip()
        b = str(r["ASLCitizenGLOSS2"]).upper().strip()
        forbidden.add((a, b))
        forbidden.add((b, a))

    rng = np.random.default_rng(seed)
    sampled = set()
    rows = []
    max_attempts = n_pairs * 200
    for _ in range(max_attempts):
        if len(rows) >= n_pairs:
            break
        i, j = rng.integers(0, len(pool), size=2)
        if i == j:
            continue
        a, b = pool[i], pool[j]
        key = (a, b) if a < b else (b, a)
        if key in sampled or (a, b) in forbidden:
            continue
        sampled.add(key)
        rows.append({"Trial": len(rows) + 1,
                     "ASLCitizenGLOSS1": a,
                     "ASLCitizenGLOSS2": b,
                     "PAIR": f"{a} / {b}"})
    if len(rows) < n_pairs:
        print(f"Warning: only sampled {len(rows)} unique non-minimal pairs (requested {n_pairs}).")
    return pd.DataFrame(rows)


def summarize_ttest(ttest_df):
    valid = ttest_df.dropna(subset=["t_stat", "p_value"])
    if valid.empty:
        return pd.DataFrame([{"n_pairs": len(ttest_df), "n_valid": 0}])
    return pd.DataFrame([{
        "n_pairs": len(ttest_df),
        "n_valid": len(valid),
        "mean_t_stat": float(valid["t_stat"].mean()),
        "median_t_stat": float(valid["t_stat"].median()),
        "mean_p_value": float(valid["p_value"].mean()),
        "median_p_value": float(valid["p_value"].median()),
        "frac_significant": float((valid["p_value"] < 0.05).mean()),
    }])


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", required=True,
                   help="Feature JSON produced by extract_features.py.")
    p.add_argument("--minimal_pairs", required=True,
                   help="CSV of minimal pairs, with columns ASLCitizenGLOSS1, ASLCitizenGLOSS2.")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--semlex_metadata", default=None,
                   help="Optional Sem-Lex split CSV. When set, restrict the feature db "
                        "to the listed videos to avoid mismatches between extracted "
                        "features and curated Sem-Lex pairs.")
    p.add_argument("--n_random_pairs", type=int, default=259,
                   help="Number of random non-minimal pairs for the control experiment.")
    p.add_argument("--random_seed", type=int, default=42)
    p.add_argument("--skip_minimal", action="store_true",
                   help="Skip the minimal-pair t-test (only run the random control).")
    p.add_argument("--skip_random", action="store_true",
                   help="Skip the random non-minimal pair control.")
    return p.parse_args()


def load_features(path):
    with open(path) as fh:
        records = json.load(fh)
    df = pd.DataFrame(records)
    df = df.apply(lambda col: col.map(lambda x: x[0] if isinstance(x, list) and len(x) == 1 else x))
    df["gloss"] = df["gloss"].astype(str).str.upper().str.strip()
    return df


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    feature_db = load_features(args.features)
    minimal_pairs = pd.read_csv(args.minimal_pairs, encoding_errors="replace")

    if args.semlex_metadata is not None:
        meta = pd.read_csv(args.semlex_metadata, encoding_errors="replace")
        valid = set(meta["Video file"].astype(str))
        valid |= {f.replace(".webm", ".mp4") for f in valid}
        feature_db = feature_db[feature_db["filename"].isin(valid)]

    if not args.skip_minimal:
        ttest = build_ttest_table(minimal_pairs, feature_db)
        ttest.to_csv(os.path.join(args.output_dir, "ttest_results.csv"), index=False)
        summarize_ttest(ttest).to_csv(
            os.path.join(args.output_dir, "ttest_summary.csv"), index=False
        )
        print("Minimal-pair t-test results written.")

    if not args.skip_random:
        random_pairs = sample_random_nonminimal_pairs(
            minimal_pairs, n_pairs=args.n_random_pairs, seed=args.random_seed,
        )
        random_pairs.to_csv(os.path.join(args.output_dir, "random_nonminimal_pairs.csv"),
                            index=False)
        random_ttest = build_ttest_table(random_pairs, feature_db)
        random_ttest.to_csv(os.path.join(args.output_dir, "random_ttest_results.csv"),
                            index=False)
        summarize_ttest(random_ttest).to_csv(
            os.path.join(args.output_dir, "random_ttest_summary.csv"), index=False
        )
        print("Random non-minimal pair control results written.")

    feature_db = feature_db.dropna()
    mean_gloss = feature_db.groupby("gloss", as_index=False)["features"].apply(
        lambda x: np.mean(list(x), axis=0).tolist()
    )
    cos_per_gloss = feature_db.groupby("gloss", as_index=False)["features"].apply(
        lambda x: mean_pairwise_cosine_similarity(list(x))
    )
    cos_per_user = feature_db.groupby("user", as_index=False)["features"].apply(
        lambda x: mean_pairwise_cosine_similarity(list(x))
    )
    mean_gloss.to_csv(os.path.join(args.output_dir, "mean_gloss_features.csv"), index=False)
    cos_per_gloss.to_csv(os.path.join(args.output_dir, "cossim_per_gloss.csv"), index=False)
    cos_per_user.to_csv(os.path.join(args.output_dir, "cossim_per_user.csv"), index=False)


if __name__ == "__main__":
    main()
