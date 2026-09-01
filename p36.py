
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
np.random.seed(42)
FIGDIR = '/home/claude/outputs/p36'
_fig_count = [0]
def savefig(name=None):
    _fig_count[0] += 1
    fname = f"{FIGDIR}/fig{_fig_count[0]}.png"
    plt.savefig(fname, dpi=110, bbox_inches="tight")
    plt.close()
    print(f"[saved figure: fig{_fig_count[0]}.png]")


from sklearn.datasets import load_breast_cancer
data = load_breast_cancer(as_frame=True)
df = data.frame
df["malignant"] = (df["target"] == 0).astype(int)  # 0 = malignant in this dataset
median_radius = df["mean radius"].median()
df["large_radius"] = (df["mean radius"] > median_radius).astype(int)

P_malignant = df["malignant"].mean()
P_large_given_malignant = df[df["malignant"]==1]["large_radius"].mean()
P_large = df["large_radius"].mean()

P_malignant_given_large_bayes = (P_large_given_malignant * P_malignant) / P_large
P_malignant_given_large_direct = df[df["large_radius"]==1]["malignant"].mean()

print(f"P(malignant) = {P_malignant:.3f}")
print(f"P(large_radius | malignant) = {P_large_given_malignant:.3f}")
print(f"P(large_radius) = {P_large:.3f}")
print(f"P(malignant | large_radius) via Bayes = {P_malignant_given_large_bayes:.3f}")
print(f"P(malignant | large_radius) direct     = {P_malignant_given_large_direct:.3f}")
