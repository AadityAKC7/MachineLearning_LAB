
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
np.random.seed(42)
FIGDIR = '/home/claude/outputs/p03'
_fig_count = [0]
def savefig(name=None):
    _fig_count[0] += 1
    fname = f"{FIGDIR}/fig{_fig_count[0]}.png"
    plt.savefig(fname, dpi=110, bbox_inches="tight")
    plt.close()
    print(f"[saved figure: fig{_fig_count[0]}.png]")


from sklearn.datasets import load_wine
from sklearn.preprocessing import MinMaxScaler, StandardScaler
data = load_wine(as_frame=True)
X = data.frame.drop(columns=["target"])
mm = MinMaxScaler().fit_transform(X)
ss = StandardScaler().fit_transform(X)
print("Original range (first 3 cols):\n", X.iloc[:, :3].agg(["min", "max"]))
print("\nMinMax scaled range (first 3 cols):\n", pd.DataFrame(mm[:, :3]).agg(["min", "max"]))
print("\nStandard scaled mean/std (first 3 cols):\n", pd.DataFrame(ss[:, :3]).agg(["mean", "std"]))
