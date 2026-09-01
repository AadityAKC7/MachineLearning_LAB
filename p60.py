
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
np.random.seed(42)
FIGDIR = '/home/claude/outputs/p60'
_fig_count = [0]
def savefig(name=None):
    _fig_count[0] += 1
    fname = f"{FIGDIR}/fig{_fig_count[0]}.png"
    plt.savefig(fname, dpi=110, bbox_inches="tight")
    plt.close()
    print(f"[saved figure: fig{_fig_count[0]}.png]")


from sklearn.neighbors import NearestNeighbors
np.random.seed(1)
n_users, n_movies = 20, 10
ratings = np.random.randint(0, 6, (n_users, n_movies))
ratings[ratings < 2] = 0  # sparsify: unrated movies = 0
movie_names = [f"Movie_{i+1}" for i in range(n_movies)]

model = NearestNeighbors(n_neighbors=4, metric="cosine").fit(ratings)
target_user = 0
distances, indices = model.kneighbors([ratings[target_user]])
neighbors = indices[0][1:]  # exclude self
print("Target user ratings:", ratings[target_user])
print("Most similar users:", neighbors)

unrated = np.where(ratings[target_user] == 0)[0]
scores = ratings[neighbors][:, unrated].mean(axis=0)
top = unrated[np.argsort(scores)[::-1][:3]]
print("Recommended movies:", [movie_names[i] for i in top])
