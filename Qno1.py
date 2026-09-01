import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["species"] = iris.target

print("First five records:\n", df.head(), "\n")
print("Dataset Shape:", df.shape, "\n")
print("Data Types:\n", df.dtypes, "\n")
print("Missing Values:\n", df.isnull().sum(), "\n")
print("Summary Statistics:\n", df.describe(), "\n")
print("Dataset Information:")
df.info()