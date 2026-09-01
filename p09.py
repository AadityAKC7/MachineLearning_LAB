
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
np.random.seed(42)
FIGDIR = '/home/claude/outputs/p09'
_fig_count = [0]
def savefig(name=None):
    _fig_count[0] += 1
    fname = f"{FIGDIR}/fig{_fig_count[0]}.png"
    plt.savefig(fname, dpi=110, bbox_inches="tight")
    plt.close()
    print(f"[saved figure: fig{_fig_count[0]}.png]")


from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
X, y = make_regression(n_samples=100, n_features=1, noise=20, random_state=7)
model = LinearRegression().fit(X, y)
pred = model.predict(X)
residuals = y - pred
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].scatter(X, y, alpha=0.6, label="data")
axes[0].plot(X, pred, color="red", label="regression line")
axes[0].set_title("Regression Line")
axes[0].legend()
axes[1].scatter(pred, residuals, alpha=0.6, color="green")
axes[1].axhline(0, color="red", linestyle="--")
axes[1].set_title("Residual Plot")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Residual")
plt.tight_layout()
savefig()
print("Residual mean:", residuals.mean(), "Residual std:", residuals.std())
