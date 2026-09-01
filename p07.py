
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
np.random.seed(42)
FIGDIR = '/home/claude/outputs/p07'
_fig_count = [0]
def savefig(name=None):
    _fig_count[0] += 1
    fname = f"{FIGDIR}/fig{_fig_count[0]}.png"
    plt.savefig(fname, dpi=110, bbox_inches="tight")
    plt.close()
    print(f"[saved figure: fig{_fig_count[0]}.png]")


from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
X, y = make_regression(n_samples=200, n_features=3, noise=10, random_state=1)
cols = ["years_experience", "education_level", "age"]
df = pd.DataFrame(X, columns=cols)
df["salary"] = y * 5000 + 50000
X_train, X_test, y_train, y_test = train_test_split(df[cols], df["salary"], test_size=0.2, random_state=42)
model = LinearRegression().fit(X_train, y_train)
print("Feature coefficients:")
for c, coef in zip(cols, model.coef_):
    print(f"  {c}: {coef:.2f}")
print("Intercept:", model.intercept_)
print("Test R2:", model.score(X_test, y_test))
