
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
np.random.seed(42)
FIGDIR = '/home/claude/outputs/p43'
_fig_count = [0]
def savefig(name=None):
    _fig_count[0] += 1
    fname = f"{FIGDIR}/fig{_fig_count[0]}.png"
    plt.savefig(fname, dpi=110, bbox_inches="tight")
    plt.close()
    print(f"[saved figure: fig{_fig_count[0]}.png]")


from sklearn.datasets import make_blobs
from scipy.cluster.hierarchy import dendrogram, linkage
X, _ = make_blobs(n_samples=30, centers=4, cluster_std=1.0, random_state=42)
Z = linkage(X, method="ward")
plt.figure(figsize=(9,5))
dendrogram(Z)
plt.title("Dendrogram (Ward Linkage)"); plt.xlabel("Sample index"); plt.ylabel("Distance")
savefig()
