
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
np.random.seed(42)
FIGDIR = '/home/claude/outputs/p05'
_fig_count = [0]
def savefig(name=None):
    _fig_count[0] += 1
    fname = f"{FIGDIR}/fig{_fig_count[0]}.png"
    plt.savefig(fname, dpi=110, bbox_inches="tight")
    plt.close()
    print(f"[saved figure: fig{_fig_count[0]}.png]")


from sklearn.datasets import load_iris
iris = load_iris(as_frame=True)
df = iris.frame
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].hist(df["petal length (cm)"], bins=15, color="steelblue")
axes[0].set_title("Histogram: Petal Length")
sc = axes[1].scatter(df["petal length (cm)"], df["petal width (cm)"], c=df["target"], cmap="viridis")
axes[1].set_title("Scatter: Petal Length vs Width")
df.iloc[:, :4].plot(kind="box", ax=axes[2])
axes[2].set_title("Box Plot: All Features")
axes[2].tick_params(axis="x", rotation=30)
plt.tight_layout()
savefig()
print("Plots created for histogram, scatter and box plot.")
