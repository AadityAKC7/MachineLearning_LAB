
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
np.random.seed(42)
FIGDIR = '/home/claude/outputs/p38'
_fig_count = [0]
def savefig(name=None):
    _fig_count[0] += 1
    fname = f"{FIGDIR}/fig{_fig_count[0]}.png"
    plt.savefig(fname, dpi=110, bbox_inches="tight")
    plt.close()
    print(f"[saved figure: fig{_fig_count[0]}.png]")


from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=1.0, random_state=42)
X[:,0] = X[:,0]*15 + 60   # annual income (k$)
X[:,1] = X[:,1]*15 + 50   # spending score
km = KMeans(n_clusters=4, n_init=10, random_state=42).fit(X)
print("Cluster centers:\n", km.cluster_centers_)
print("Cluster sizes:", np.bincount(km.labels_))
