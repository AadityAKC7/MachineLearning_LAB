
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
np.random.seed(42)
FIGDIR = '/home/claude/outputs/p42'
_fig_count = [0]
def savefig(name=None):
    _fig_count[0] += 1
    fname = f"{FIGDIR}/fig{_fig_count[0]}.png"
    plt.savefig(fname, dpi=110, bbox_inches="tight")
    plt.close()
    print(f"[saved figure: fig{_fig_count[0]}.png]")


from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
X, _ = make_blobs(n_samples=200, centers=4, cluster_std=1.0, random_state=42)

def bisecting_kmeans(X, k):
    clusters = [np.arange(len(X))]
    while len(clusters) < k:
        sizes = [len(c) for c in clusters]
        idx = np.argmax(sizes)
        biggest = clusters.pop(idx)
        km = KMeans(n_clusters=2, n_init=10, random_state=42).fit(X[biggest])
        clusters.append(biggest[km.labels_ == 0])
        clusters.append(biggest[km.labels_ == 1])
    labels = np.zeros(len(X), dtype=int)
    for i, c in enumerate(clusters):
        labels[c] = i
    return labels

labels = bisecting_kmeans(X, 4)
print("Divisive cluster sizes:", np.bincount(labels))
plt.figure(figsize=(6,5))
plt.scatter(X[:,0], X[:,1], c=labels, cmap="tab10")
plt.title("Divisive Hierarchical Clustering (Bisecting K-Means)")
savefig()
