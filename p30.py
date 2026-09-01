
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
np.random.seed(42)
FIGDIR = '/home/claude/outputs/p30'
_fig_count = [0]
def savefig(name=None):
    _fig_count[0] += 1
    fname = f"{FIGDIR}/fig{_fig_count[0]}.png"
    plt.savefig(fname, dpi=110, bbox_inches="tight")
    plt.close()
    print(f"[saved figure: fig{_fig_count[0]}.png]")


from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
gnb = GaussianNB().fit(X_train, y_train)
print("GaussianNB (Iris) accuracy:", accuracy_score(y_test, gnb.predict(X_test)))

Xc = np.random.randint(0, 5, (300, 10))
yc = (Xc.sum(axis=1) > 22).astype(int)
Xc_train, Xc_test, yc_train, yc_test = train_test_split(Xc, yc, test_size=0.2, random_state=42)
mnb = MultinomialNB().fit(Xc_train, yc_train)
print("MultinomialNB (synthetic counts) accuracy:", accuracy_score(yc_test, mnb.predict(Xc_test)))
