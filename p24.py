
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
np.random.seed(42)
FIGDIR = '/home/claude/outputs/p24'
_fig_count = [0]
def savefig(name=None):
    _fig_count[0] += 1
    fname = f"{FIGDIR}/fig{_fig_count[0]}.png"
    plt.savefig(fname, dpi=110, bbox_inches="tight")
    plt.close()
    print(f"[saved figure: fig{_fig_count[0]}.png]")


from sklearn.datasets import load_digits
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
X, y = load_digits(return_X_y=True)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler().fit(X_train)
X_train, X_val = scaler.transform(X_train), scaler.transform(X_val)
classes = np.unique(y_train)

mlp = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=1, warm_start=True, random_state=42)
train_acc, val_acc = [], []
for epoch in range(60):
    mlp.partial_fit(X_train, y_train, classes=classes)
    train_acc.append(accuracy_score(y_train, mlp.predict(X_train)))
    val_acc.append(accuracy_score(y_val, mlp.predict(X_val)))

plt.figure(figsize=(7,4))
plt.plot(train_acc, label="Train Accuracy")
plt.plot(val_acc, label="Validation Accuracy")
plt.legend(); plt.title("Overfitting: Train vs Validation Accuracy"); plt.xlabel("Epoch")
savefig()
print("Final train acc:", train_acc[-1], "Final val acc:", val_acc[-1])
