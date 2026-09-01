
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
np.random.seed(42)
FIGDIR = '/home/claude/outputs/p45'
_fig_count = [0]
def savefig(name=None):
    _fig_count[0] += 1
    fname = f"{FIGDIR}/fig{_fig_count[0]}.png"
    plt.savefig(fname, dpi=110, bbox_inches="tight")
    plt.close()
    print(f"[saved figure: fig{_fig_count[0]}.png]")


from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
X, y = load_digits(return_X_y=True)
Xs = StandardScaler().fit_transform(X)
X_pca = PCA(n_components=2).fit_transform(Xs)

fig, axes = plt.subplots(1, 2, figsize=(12,5))
axes[0].scatter(X[:,20], X[:,36], c=y, cmap="tab10", s=10)
axes[0].set_title("Before PCA (2 raw pixel features)")
sc = axes[1].scatter(X_pca[:,0], X_pca[:,1], c=y, cmap="tab10", s=10)
axes[1].set_title("After PCA (2 principal components)")
plt.colorbar(sc, ax=axes[1], label="Digit class")
savefig()
