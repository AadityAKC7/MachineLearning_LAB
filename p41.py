
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
np.random.seed(42)
FIGDIR = '/home/claude/outputs/p41'
_fig_count = [0]
def savefig(name=None):
    _fig_count[0] += 1
    fname = f"{FIGDIR}/fig{_fig_count[0]}.png"
    plt.savefig(fname, dpi=110, bbox_inches="tight")
    plt.close()
    print(f"[saved figure: fig{_fig_count[0]}.png]")


from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
X, _ = make_blobs(n_samples=200, centers=4, cluster_std=1.0, random_state=42)
agg = AgglomerativeClustering(n_clusters=4, linkage="ward").fit(X)
print("Cluster sizes:", np.bincount(agg.labels_))
plt.figure(figsize=(6,5))
plt.scatter(X[:,0], X[:,1], c=agg.labels_, cmap="tab10")
plt.title("Agglomerative Hierarchical Clustering")
savefig()
