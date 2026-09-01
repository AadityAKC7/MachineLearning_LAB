
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
np.random.seed(42)
FIGDIR = '/home/claude/outputs/p29'
_fig_count = [0]
def savefig(name=None):
    _fig_count[0] += 1
    fname = f"{FIGDIR}/fig{_fig_count[0]}.png"
    plt.savefig(fname, dpi=110, bbox_inches="tight")
    plt.close()
    print(f"[saved figure: fig{_fig_count[0]}.png]")


from sklearn.datasets import make_blobs
from sklearn.svm import SVC
X, y = make_blobs(n_samples=60, centers=2, cluster_std=1.2, random_state=6)
clf = SVC(kernel="linear", C=1000).fit(X, y)
plt.figure(figsize=(7,5))
plt.scatter(X[:,0], X[:,1], c=y, cmap="coolwarm")
ax = plt.gca()
xlim = ax.get_xlim(); ylim = ax.get_ylim()
xx = np.linspace(xlim[0], xlim[1], 30); yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()]).reshape(XX.shape)
ax.contour(XX, YY, Z, colors="k", levels=[-1, 0, 1], linestyles=["--", "-", "--"])
ax.scatter(clf.support_vectors_[:,0], clf.support_vectors_[:,1], s=140,
           facecolors="none", edgecolors="green", linewidths=2, label="Support Vectors")
plt.legend(); plt.title("SVM Margin and Support Vectors")
savefig()
print("Number of support vectors:", len(clf.support_vectors_))
