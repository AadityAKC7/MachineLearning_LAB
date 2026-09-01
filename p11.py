
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
np.random.seed(42)
FIGDIR = '/home/claude/outputs/p11'
_fig_count = [0]
def savefig(name=None):
    _fig_count[0] += 1
    fname = f"{FIGDIR}/fig{_fig_count[0]}.png"
    plt.savefig(fname, dpi=110, bbox_inches="tight")
    plt.close()
    print(f"[saved figure: fig{_fig_count[0]}.png]")


def perceptron_train(X, y, lr=0.1, epochs=20):
    w = np.zeros(X.shape[1]); b = 0.0
    for _ in range(epochs):
        for xi, target in zip(X, y):
            pred = 1 if np.dot(w, xi) + b > 0 else 0
            update = lr * (target - pred)
            w += update * xi
            b += update
    return w, b

X = np.array([[0,0],[0,1],[1,0],[1,1]])
y_and = np.array([0,0,0,1])
y_or  = np.array([0,1,1,1])

for name, y in [("AND", y_and), ("OR", y_or)]:
    w, b = perceptron_train(X, y)
    preds = [1 if np.dot(w, xi) + b > 0 else 0 for xi in X]
    print(f"{name} gate -> weights={w}, bias={b:.2f}, predictions={preds}, target={list(y)}")
