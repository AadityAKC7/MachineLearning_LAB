
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
np.random.seed(42)
FIGDIR = '/home/claude/outputs/p26'
_fig_count = [0]
def savefig(name=None):
    _fig_count[0] += 1
    fname = f"{FIGDIR}/fig{_fig_count[0]}.png"
    plt.savefig(fname, dpi=110, bbox_inches="tight")
    plt.close()
    print(f"[saved figure: fig{_fig_count[0]}.png]")


from sklearn.tree import DecisionTreeClassifier, plot_tree
n = 500
income = np.random.randint(15000, 120000, n)
credit_score = np.random.randint(300, 850, n)
loan_amount = np.random.randint(1000, 50000, n)
approved = ((credit_score > 600) & (income > 30000) & (loan_amount < income * 0.5)).astype(int)
X = np.column_stack([income, credit_score, loan_amount])
clf = DecisionTreeClassifier(max_depth=3, random_state=42).fit(X, approved)
plt.figure(figsize=(14, 7))
plot_tree(clf, feature_names=["income", "credit_score", "loan_amount"],
          class_names=["Rejected", "Approved"], filled=True, fontsize=9)
savefig()
print("Tree depth:", clf.get_depth(), "Number of leaves:", clf.get_n_leaves())
