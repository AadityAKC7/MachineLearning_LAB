
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
np.random.seed(42)
FIGDIR = '/home/claude/outputs/p21'
_fig_count = [0]
def savefig(name=None):
    _fig_count[0] += 1
    fname = f"{FIGDIR}/fig{_fig_count[0]}.png"
    plt.savefig(fname, dpi=110, bbox_inches="tight")
    plt.close()
    print(f"[saved figure: fig{_fig_count[0]}.png]")


from sklearn.datasets import load_digits
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler().fit(X_train)
X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)

configs = [
    ("SGD (no momentum)", dict(solver="sgd", momentum=0.0)),
    ("SGD + Momentum", dict(solver="sgd", momentum=0.9)),
    ("Adam", dict(solver="adam")),
    ("LBFGS (batch GD)", dict(solver="lbfgs")),
]
for name, kw in configs:
    mlp = MLPClassifier(hidden_layer_sizes=(64,), max_iter=300, random_state=42, **kw).fit(X_train, y_train)
    acc = accuracy_score(y_test, mlp.predict(X_test))
    print(f"{name:20s} -> accuracy={acc:.4f}")
