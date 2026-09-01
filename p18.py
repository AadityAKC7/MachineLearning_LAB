
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
np.random.seed(42)
FIGDIR = '/home/claude/outputs/p18'
_fig_count = [0]
def savefig(name=None):
    _fig_count[0] += 1
    fname = f"{FIGDIR}/fig{_fig_count[0]}.png"
    plt.savefig(fname, dpi=110, bbox_inches="tight")
    plt.close()
    print(f"[saved figure: fig{_fig_count[0]}.png]")


x = np.linspace(-10, 10, 400)
sigmoid = 1 / (1 + np.exp(-x))
tanh = np.tanh(x)
relu = np.maximum(0, x)
d_sigmoid = sigmoid * (1 - sigmoid)
d_tanh = 1 - tanh**2
d_relu = (x > 0).astype(float)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(x, sigmoid, label="Sigmoid")
axes[0].plot(x, tanh, label="Tanh")
axes[0].plot(x, relu, label="ReLU")
axes[0].set_title("Activation Functions"); axes[0].legend()
axes[1].plot(x, d_sigmoid, label="Sigmoid'")
axes[1].plot(x, d_tanh, label="Tanh'")
axes[1].plot(x, d_relu, label="ReLU'")
axes[1].set_title("Gradients"); axes[1].legend()
plt.tight_layout()
savefig()
print("Plotted activation functions and their derivatives.")
