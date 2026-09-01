
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
np.random.seed(42)
FIGDIR = '/home/claude/outputs/p37'
_fig_count = [0]
def savefig(name=None):
    _fig_count[0] += 1
    fname = f"{FIGDIR}/fig{_fig_count[0]}.png"
    plt.savefig(fname, dpi=110, bbox_inches="tight")
    plt.close()
    print(f"[saved figure: fig{_fig_count[0]}.png]")


np.random.seed(0)
N = 100000
rain = np.random.rand(N) < 0.2  # P(Rain) = 0.2

sprinkler = np.zeros(N, dtype=bool)
sprinkler[rain] = np.random.rand(rain.sum()) < 0.01     # P(Sprinkler | Rain) = 0.01
sprinkler[~rain] = np.random.rand((~rain).sum()) < 0.4   # P(Sprinkler | ~Rain) = 0.4

wet = np.zeros(N, dtype=bool)
both = rain & sprinkler
only_rain = rain & ~sprinkler
only_sprinkler = ~rain & sprinkler
neither = ~rain & ~sprinkler
wet[both] = np.random.rand(both.sum()) < 0.99
wet[only_rain] = np.random.rand(only_rain.sum()) < 0.8
wet[only_sprinkler] = np.random.rand(only_sprinkler.sum()) < 0.9
wet[neither] = np.random.rand(neither.sum()) < 0.0

print("Simulated P(Rain):", rain.mean())
print("Simulated P(Sprinkler):", sprinkler.mean())
print("Simulated P(WetGrass):", wet.mean())
print("Simulated P(Rain | WetGrass):", rain[wet].mean())
