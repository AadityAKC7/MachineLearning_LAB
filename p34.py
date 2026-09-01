
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
np.random.seed(42)
FIGDIR = '/home/claude/outputs/p34'
_fig_count = [0]
def savefig(name=None):
    _fig_count[0] += 1
    fname = f"{FIGDIR}/fig{_fig_count[0]}.png"
    plt.savefig(fname, dpi=110, bbox_inches="tight")
    plt.close()
    print(f"[saved figure: fig{_fig_count[0]}.png]")


from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler().fit(X_train)
X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)

for metric in ["euclidean", "manhattan"]:
    for k in [1, 3, 5, 7, 9, 15]:
        knn = KNeighborsClassifier(n_neighbors=k, metric=metric).fit(X_train, y_train)
        print(f"metric={metric:10s} k={k:2d} -> accuracy={knn.score(X_test, y_test):.4f}")
