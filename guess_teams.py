#!/usr/bin/env python3
"""
Cluster authors into teams using ILP + PuLP, from commits.csv.
"""

import pandas as pd, numpy as np, pulp, argparse, math
from sklearn.metrics.pairwise import cosine_similarity

# -------- Parameters (edit here) --------
CSV_FILE = "commits.csv"
SINCE_DAYS = 730  # ignore commits older than this
HALF_LIFE = 180  # weigh down old commits
TOP_AUTHORS = 25  # only consider this many authors (performance)
TOP_REPOS = 35  # only consider this many repos (performance)
REPO_NORM = "idf"  # "none","colsum","idf"
K = 3  # clusters (2,3,4)
MIN_SIZE = 3  # minimum number of people on a team
MAX_SIZE = 8  # maximum number of people on a team
OUT_FILE = "assignments.csv"

# ----------------------------------------

def build_matrix(df):
    # Parse timestamps as UTC to avoid mixed tz issues
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])

    # Numeric columns
    df["lines_added"] = pd.to_numeric(df["lines_added"], errors="coerce").fillna(0)
    df["lines_removed"] = pd.to_numeric(df["lines_removed"], errors="coerce").fillna(0)

    # Time filter (last SINCE_DAYS relative to max timestamp)
    tmax = df["timestamp"].max()
    cutoff = tmax - pd.Timedelta(days=SINCE_DAYS)
    df = df[df["timestamp"] >= cutoff]
    if df.empty:
        raise SystemExit("No rows after time filter.")

    # Recency decay (days, using total_seconds to stay tz-safe)
    age_days = (tmax - df["timestamp"]).dt.total_seconds() / 86400.0
    age_days = age_days.clip(lower=0)
    decay = np.power(2.0, -age_days / float(HALF_LIFE))
    df["w"] = (df["lines_added"].clip(lower=0) + df["lines_removed"].clip(lower=0)) * decay.to_numpy()

    # Top authors/repos
    A = df.groupby("author")["w"].sum().sort_values(ascending=False).head(TOP_AUTHORS).index.tolist()
    R = df.groupby("repo")["w"].sum().sort_values(ascending=False).head(TOP_REPOS).index.tolist()
    df = df[df["author"].isin(A) & df["repo"].isin(R)]
    if df.empty:
        raise SystemExit("No rows after top-k selection.")

    # Author x Repo matrix
    P = df.pivot_table(index="author", columns="repo", values="w", aggfunc="sum", fill_value=0.0) \
        .reindex(index=A, columns=R, fill_value=0.0)
    M = P.to_numpy(float)

    # Repo downweighting
    if REPO_NORM == "colsum":
        M = M / (M.sum(axis=0, keepdims=True) + 1e-12)
    elif REPO_NORM == "idf":
        touched = (M > 0).sum(axis=0)
        n = M.shape[0]
        idf = np.log((1 + n) / (1 + touched)) + 1.0
        M = M * idf

    # Row L1 normalization
    M = M / (M.sum(axis=1, keepdims=True) + 1e-12)
    return P.index.tolist(), P.columns.tolist(), M


def load_pairs(path):
    if not path: return []
    pairs = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#"): continue
        a, b = [t.strip() for t in line.split(",")]
        pairs.append((a, b))
    return pairs


def ilp_partition(authors, M, k, min_size, max_size, must_pairs, cannot_pairs):
    n = len(authors)
    C = range(k)
    W = cosine_similarity(M)
    prob = pulp.LpProblem("partition", pulp.LpMaximize)
    x = pulp.LpVariable.dicts("x", (range(n), C), 0, 1, cat="Binary")
    y = {(i, j, c): pulp.LpVariable(f"y_{i}_{j}_{c}", 0, 1, cat="Binary") for i in range(n) for j in range(i + 1, n) for
         c in C}

    for i in range(n): prob += pulp.lpSum(x[i][c] for c in C) == 1
    for (i, j, c), var in y.items():
        prob += var <= x[i][c]
        prob += var <= x[j][c]
        prob += var >= x[i][c] + x[j][c] - 1

    minc = min_size
    maxc = max_size
    for c in C:
        prob += pulp.lpSum(x[i][c] for i in range(n)) <= maxc
        prob += pulp.lpSum(x[i][c] for i in range(n)) >= minc

    name2i = {a: i for i, a in enumerate(authors)}
    for a, b in must_pairs:
        if a in name2i and b in name2i:
            i, j = name2i[a], name2i[b]
            for c in C: prob += x[i][c] - x[j][c] == 0
    for a, b in cannot_pairs:
        if a in name2i and b in name2i:
            i, j = name2i[a], name2i[b]
            if i > j: i, j = j, i
            prob += pulp.lpSum(y[(i, j, c)] for c in C) == 0

    prob += pulp.lpSum(W[i, j] * y[(i, j, c)] for (i, j, c) in y)
    prob.solve(pulp.PULP_CBC_CMD(msg=False, threads=12))
    labels = np.zeros(n, dtype=int)
    for i in range(n):
        for c in C:
            if pulp.value(x[i][c]) > 0.5: labels[i] = c; break
    return labels, pulp.value(prob.objective)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--must-link")
    ap.add_argument("--cannot-link")
    args = ap.parse_args()
    df = pd.read_csv(CSV_FILE)
    authors, repos, M = build_matrix(df)
    must = load_pairs(args.must_link)
    cannot = load_pairs(args.cannot_link)

    print(
        f"Targeting {K} teams. Matrix shape (authors x repos): {M.shape}. "
        f"Must-link: {len(must)}, "
        f"Cannot-link: {len(cannot)}. "
        f"Starting ILP computation...")
    labels, obj = ilp_partition(authors, M, K, MIN_SIZE, MAX_SIZE, must, cannot)
    out = pd.DataFrame({"author": authors, "cluster": labels})
    out.to_csv(OUT_FILE, index=False)
    for c in sorted(set(labels)):
        A = [a for a, i in zip(authors, labels) if i == c]
        print(f"Cluster {c} ({len(A)}):", ", ".join(A))
    print("Saved", OUT_FILE)

def find_best_cluster_size(authors, M, k, must, cannot):
    n = len(authors)
    base_size = int(math.ceil(n / k))
    best_labels = None
    best_obj = float('-inf')
    best_size = None

    for size in range(base_size, base_size + 3):
        labels, obj = ilp_partition(authors, M, k, size, size, must, cannot)
        if obj > best_obj:
            best_obj = obj
            best_labels = labels
            best_size = size
    return best_labels, best_size, best_obj


if __name__ == "__main__": main()
