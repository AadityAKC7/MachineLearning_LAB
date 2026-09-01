
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
np.random.seed(42)
FIGDIR = '/home/claude/outputs/p02'
_fig_count = [0]
def savefig(name=None):
    _fig_count[0] += 1
    fname = f"{FIGDIR}/fig{_fig_count[0]}.png"
    plt.savefig(fname, dpi=110, bbox_inches="tight")
    plt.close()
    print(f"[saved figure: fig{_fig_count[0]}.png]")


n = 20
df = pd.DataFrame({
    "age": np.random.randint(18, 60, n).astype(float),
    "income": np.random.randint(20000, 90000, n).astype(float),
    "city": np.random.choice(["Kathmandu", "Pokhara", "Biratnagar"], n)
})
mask_num = np.random.choice([True, False], n, p=[0.15, 0.85])
df.loc[mask_num, "age"] = np.nan
mask_cat = np.random.choice([True, False], n, p=[0.15, 0.85])
df.loc[mask_cat, "city"] = np.nan
print("Before:\n", df.isnull().sum())
df["age"] = df["age"].fillna(df["age"].mean())
df["income"] = df["income"].fillna(df["income"].mean())
df["city"] = df["city"].fillna(df["city"].mode()[0])
print("\nAfter imputation:\n", df.isnull().sum())
encoded = pd.get_dummies(df, columns=["city"], prefix="city")
print("\nEncoded columns:\n", encoded.columns.tolist())
print(encoded.head())
