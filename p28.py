
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
np.random.seed(42)
FIGDIR = '/home/claude/outputs/p28'
_fig_count = [0]
def savefig(name=None):
    _fig_count[0] += 1
    fname = f"{FIGDIR}/fig{_fig_count[0]}.png"
    plt.savefig(fname, dpi=110, bbox_inches="tight")
    plt.close()
    print(f"[saved figure: fig{_fig_count[0]}.png]")


from sklearn.datasets import make_moons
from sklearn.svm import SVC
X, y = make_moons(n_samples=200, noise=0.2, random_state=42)
kernels = ["linear", "poly", "rbf"]
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
xx, yy = np.meshgrid(np.linspace(X[:,0].min()-.5, X[:,0].max()+.5, 200),
                      np.linspace(X[:,1].min()-.5, X[:,1].max()+.5, 200))
for ax, kernel in zip(axes, kernels):
    clf = SVC(kernel=kernel, gamma="scale").fit(X, y)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
    ax.scatter(X[:,0], X[:,1], c=y, cmap="coolwarm", edgecolors="k")
    ax.set_title(f"kernel={kernel}, acc={clf.score(X,y):.2f}")
plt.tight_layout()
savefig()
