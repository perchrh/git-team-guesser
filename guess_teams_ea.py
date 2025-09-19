import pandas as pd
import random
from deap import base, creator, tools, algorithms

# Load the commits.csv file into a DataFrame
df = pd.read_csv('commits.csv', parse_dates=['timestamp'])

# Build author -> set of repos mapping
author_repos = df.groupby('author')['repo'].apply(set).to_dict()
authors = list(author_repos.keys())
n_authors = len(authors)
n_clusters = 2  # or 2

import numpy as np

# Parse timestamps and compute age in days
max_time = df['timestamp'].max()
df['age'] = max_time - df['timestamp']

# Decay parameter (tune as needed)
# Set decay_lambda so that weight at 1000 days is 0.0001
decay_lambda = -np.log(0.0001) / 1000  # ≈ 0.000945

# Compute decayed weights for each commit
# Use age in days for decay calculation
df['decay_weight'] = np.exp(-decay_lambda * (df['age'] / np.timedelta64(1, 'D')))

# Compute per-commit weight: lines_added + lines_removed, weighted by decay
df['commit_weight'] = (df['lines_added'] + df['lines_removed']) * df['decay_weight']

# Build author-repo weighted table
author_repo_decay = df.groupby(['author', 'repo'])['commit_weight'].sum().unstack(fill_value=0)

# Compute number of commits per author
author_commit_counts = df['author'].value_counts().to_dict()
penalty_lambda = 0.005  # Tune this parameter
min_cluster_size = 3

def fitness(individual):
    clusters = [[] for _ in range(n_clusters)]
    for idx, cluster_id in enumerate(individual):
        clusters[cluster_id].append(authors[idx])
    weighted_overlap = 0
    penalty = 0
    for cluster in clusters:
        if len(cluster) < min_cluster_size:
            return (float('inf'),) # infinite penalty for small clusters
        for i in range(len(cluster)):
            for j in range(i + 1, len(cluster)):
                a1, a2 = cluster[i], cluster[j]
                # Overlap as before
                shared_repos = set(author_repo_decay.columns[author_repo_decay.loc[a1] > 0]) & \
                               set(author_repo_decay.columns[author_repo_decay.loc[a2] > 0])
                for repo in shared_repos:
                    weighted_overlap += min(author_repo_decay.loc[a1, repo], author_repo_decay.loc[a2, repo])
                # Penalize productive authors in same cluster
                penalty += penalty_lambda * author_commit_counts[a1] * author_commit_counts[a2]
    return (weighted_overlap + penalty,)



# DEAP setup
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attr_cluster", random.randrange, n_clusters)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_cluster, n_authors)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", fitness)
toolbox.register("mate", tools.cxUniform, indpb=0.2)
toolbox.register("mutate", tools.mutUniformInt, low=0, up=n_clusters - 1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=5)

# Run GA
pop = toolbox.population(n=500)
algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.1, ngen=100, verbose=True)

# Get best clustering
best = tools.selBest(pop, 1)[0]
clusters = [[] for _ in range(n_clusters)]
for idx, cluster_id in enumerate(best):
    clusters[cluster_id].append(authors[idx])

for i, cluster in enumerate(clusters, 1):
    print(f"Cluster {i}: {', '.join(cluster)}")
