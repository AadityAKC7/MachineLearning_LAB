
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
np.random.seed(42)
FIGDIR = '/home/claude/outputs/p46'
_fig_count = [0]
def savefig(name=None):
    _fig_count[0] += 1
    fname = f"{FIGDIR}/fig{_fig_count[0]}.png"
    plt.savefig(fname, dpi=110, bbox_inches="tight")
    plt.close()
    print(f"[saved figure: fig{_fig_count[0]}.png]")


from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.stats import mode

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = LogisticRegression(max_iter=1000).fit(X_train, y_train)
sup_acc = accuracy_score(y_test, clf.predict(X_test))

km = KMeans(n_clusters=3, n_init=10, random_state=42).fit(X)
mapped = np.zeros_like(km.labels_)
for cluster in range(3):
    mask = km.labels_ == cluster
    mapped[mask] = mode(y[mask], keepdims=True).mode[0]
unsup_acc = accuracy_score(y, mapped)

print("Supervised (Logistic Regression) test accuracy:", sup_acc)
print("Unsupervised (K-Means, label-aligned) accuracy:", unsup_acc)
