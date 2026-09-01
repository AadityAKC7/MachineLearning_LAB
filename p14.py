
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
np.random.seed(42)
FIGDIR = '/home/claude/outputs/p14'
_fig_count = [0]
def savefig(name=None):
    _fig_count[0] += 1
    fname = f"{FIGDIR}/fig{_fig_count[0]}.png"
    plt.savefig(fname, dpi=110, bbox_inches="tight")
    plt.close()
    print(f"[saved figure: fig{_fig_count[0]}.png]")


from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
X = np.random.uniform(0, 100, (2000, 2))
y = X[:, 0] + X[:, 1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
mlp = MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=2000, random_state=42).fit(X_train, y_train)
pred = mlp.predict(X_test)
print("R2 score on addition task:", r2_score(y_test, pred))
print("Sample predictions:", np.round(pred[:5], 2), "vs actual:", np.round(y_test[:5], 2))
