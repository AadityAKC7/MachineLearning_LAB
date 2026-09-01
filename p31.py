
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
np.random.seed(42)
FIGDIR = '/home/claude/outputs/p31'
_fig_count = [0]
def savefig(name=None):
    _fig_count[0] += 1
    fname = f"{FIGDIR}/fig{_fig_count[0]}.png"
    plt.savefig(fname, dpi=110, bbox_inches="tight")
    plt.close()
    print(f"[saved figure: fig{_fig_count[0]}.png]")


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

texts = [
    "I love this movie, it was fantastic", "Amazing acting and great story",
    "Best film I have seen this year", "What a wonderful and touching experience",
    "Absolutely brilliant, highly recommend", "I hated this movie, it was terrible",
    "Worst film ever, total waste of time", "Boring and poorly written plot",
    "I did not enjoy it at all", "Awful acting, very disappointing",
]
labels = [1,1,1,1,1,0,0,0,0,0]
vec = CountVectorizer()
X = vec.fit_transform(texts)
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3, random_state=42)
model = MultinomialNB().fit(X_train, y_train)
print("Test Accuracy:", accuracy_score(y_test, model.predict(X_test)))
new = ["This was a fantastic and touching film", "Terrible, boring, waste of my time"]
print("New predictions:", model.predict(vec.transform(new)), "(1=positive, 0=negative)")
