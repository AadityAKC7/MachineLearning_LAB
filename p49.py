
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
np.random.seed(42)
FIGDIR = '/home/claude/outputs/p49'
_fig_count = [0]
def savefig(name=None):
    _fig_count[0] += 1
    fname = f"{FIGDIR}/fig{_fig_count[0]}.png"
    plt.savefig(fname, dpi=110, bbox_inches="tight")
    plt.close()
    print(f"[saved figure: fig{_fig_count[0]}.png]")


from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score
X, y = make_classification(n_samples=2000, n_features=10, weights=[0.95, 0.05], random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
clf = LogisticRegression(max_iter=1000).fit(X_train, y_train)
pred = clf.predict(X_test)
baseline_acc = 1 - y_test.mean()
print("Class distribution in test set:", np.bincount(y_test))
print("Majority-class-only baseline accuracy:", baseline_acc)
print("Model accuracy:", accuracy_score(y_test, pred))
print("Model recall (minority class):", recall_score(y_test, pred))
print("Model precision (minority class):", precision_score(y_test, pred))
