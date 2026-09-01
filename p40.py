
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
np.random.seed(42)
FIGDIR = '/home/claude/outputs/p40'
_fig_count = [0]
def savefig(name=None):
    _fig_count[0] += 1
    fname = f"{FIGDIR}/fig{_fig_count[0]}.png"
    plt.savefig(fname, dpi=110, bbox_inches="tight")
    plt.close()
    print(f"[saved figure: fig{_fig_count[0]}.png]")


from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=1.0, random_state=42)
X[:,0] = X[:,0]*15 + 60
X[:,1] = X[:,1]*15 + 50
km = KMeans(n_clusters=4, n_init=10, random_state=42).fit(X)
plt.figure(figsize=(7,5))
plt.scatter(X[:,0], X[:,1], c=km.labels_, cmap="tab10", alpha=0.7)
plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1], c="black", marker="X", s=200, label="Centroids")
plt.xlabel("Annual Income (k$)"); plt.ylabel("Spending Score"); plt.legend(); plt.title("K-Means Clustering")
savefig()
