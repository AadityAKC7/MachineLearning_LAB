
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
np.random.seed(42)
FIGDIR = '/home/claude/outputs/p25'
_fig_count = [0]
def savefig(name=None):
    _fig_count[0] += 1
    fname = f"{FIGDIR}/fig{_fig_count[0]}.png"
    plt.savefig(fname, dpi=110, bbox_inches="tight")
    plt.close()
    print(f"[saved figure: fig{_fig_count[0]}.png]")


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
n = 500
income = np.random.randint(15000, 120000, n)
credit_score = np.random.randint(300, 850, n)
loan_amount = np.random.randint(1000, 50000, n)
approved = ((credit_score > 600) & (income > 30000) & (loan_amount < income * 0.5)).astype(int)
noise = np.random.rand(n) < 0.05
approved = np.where(noise, 1 - approved, approved)
X = np.column_stack([income, credit_score, loan_amount])
X_train, X_test, y_train, y_test = train_test_split(X, approved, test_size=0.2, random_state=42)
clf = DecisionTreeClassifier(max_depth=5, random_state=42).fit(X_train, y_train)
print("Test Accuracy:", accuracy_score(y_test, clf.predict(X_test)))
