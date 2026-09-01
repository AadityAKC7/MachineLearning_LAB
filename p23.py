
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
np.random.seed(42)
FIGDIR = '/home/claude/outputs/p23'
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
X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler().fit(X_train)
X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)

plt.figure(figsize=(7,4))
for lr in [0.001, 0.01, 0.1, 1.0]:
    mlp = MLPClassifier(hidden_layer_sizes=(64,), solver="sgd", learning_rate_init=lr,
                         max_iter=200, random_state=42).fit(X_train, y_train)
    plt.plot(mlp.loss_curve_, label=f"lr={lr}")
    print(f"learning_rate={lr} -> final loss={mlp.loss_curve_[-1]:.4f}, test acc={mlp.score(X_test, y_test):.4f}")
plt.legend(); plt.title("Loss Curves for Different Learning Rates"); plt.xlabel("Iteration"); plt.ylabel("Loss")
savefig()
