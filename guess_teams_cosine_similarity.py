import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
from scipy.spatial.distance import cosine

# 1. Load data
df = pd.read_csv('commits.csv', parse_dates=['timestamp'])

# 2. Time-decay weighting
now = df['timestamp'].max()
decay = lambda t: np.exp(-(now - t).days / 90)  # 3 month half-life
df['decay_weight'] = df['timestamp'].apply(decay)

# 3. Weighted contribution: (lines_added + lines_removed) * decay
df['weight'] = (df['lines_added'] + df['lines_removed']) * df['decay_weight']

# 4. Build author-repo matrix with weights
pivot = df.pivot_table(
    index='author', columns='repo',
    values='weight', aggfunc='sum', fill_value=0
)

# 5. Normalize for productivity (row-wise L2 norm)
features = normalize(pivot, axis=1)

# 6. Compute pairwise similarities
authors = pivot.index.tolist()
similarity = np.zeros((len(authors), len(authors)))
for i in range(len(authors)):
    for j in range(len(authors)):
        similarity[i, j] = 1 - cosine(features[i], features[j])


# 7. Custom grouping: greedy assignment to 2 or 3 clusters
def group_authors(similarity, n_clusters):
    clusters = [[] for _ in range(n_clusters)]
    assigned = set()
    # Seed clusters with the first n_clusters unique authors
    for i in range(n_clusters):
        clusters[i].append(authors[i])
        assigned.add(authors[i])
    # Assign remaining authors to the most similar cluster
    for idx, author in enumerate(authors):
        if author in assigned:
            continue
        sims = [np.mean([similarity[idx, authors.index(a)] for a in cluster]) if cluster else 0 for cluster in clusters]
        best = np.argmax(sims)
        clusters[best].append(author)
        assigned.add(author)
    # Ensure min cluster size
    for cluster in clusters:
        if len(cluster) < 3:
            return None
        if len(cluster) > 6:
            return None
    return clusters


print(f"Number of authors: {len(authors)}")
print(f"Authors: {authors}")

# Try 2 and 3 clusters
for n in [2, 3]:
    clusters = group_authors(similarity, n)
    if clusters:
        for number, cluster in enumerate(clusters, 1):
            print(f"Cluster ({number}): {', '.join(cluster)}")
