
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
np.random.seed(42)
FIGDIR = '/home/claude/outputs/p59'
_fig_count = [0]
def savefig(name=None):
    _fig_count[0] += 1
    fname = f"{FIGDIR}/fig{_fig_count[0]}.png"
    plt.savefig(fname, dpi=110, bbox_inches="tight")
    plt.close()
    print(f"[saved figure: fig{_fig_count[0]}.png]")


from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
X, _ = make_blobs(n_samples=400, centers=5, cluster_std=1.0, random_state=42)
X[:,0] = X[:,0]*12 + 55
X[:,1] = X[:,1]*12 + 50
km = KMeans(n_clusters=5, n_init=10, random_state=42).fit(X)
df = pd.DataFrame(X, columns=["annual_income","spending_score"])
df["segment"] = km.labels_
summary = df.groupby("segment").mean()
print("Segment profiles (mean income / spending score):\n", summary)
plt.figure(figsize=(7,5))
plt.scatter(X[:,0], X[:,1], c=km.labels_, cmap="tab10", alpha=0.7)
plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1], c="black", marker="X", s=200)
plt.xlabel("Annual Income"); plt.ylabel("Spending Score"); plt.title("Customer Segments")
savefig()
