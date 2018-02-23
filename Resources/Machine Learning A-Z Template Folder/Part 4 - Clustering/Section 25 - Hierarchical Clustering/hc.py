# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import the mall dataset with pandas
dataset = pd.read_csv("mall.csv")
X = dataset.iloc[:, [3, 4]].values

# using dendogram to find optimal number of cluster
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method="ward"))
plt.title("Dendrogram")
plt.xlabel("Customers")
plt.ylabel("Euclidean distance")
plt.show()

# Fitting hierarical clustering
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=5, affinity="euclidean", linkage="ward")
y_hc = hc.fit_predict(X)


# Visualizing the clusters
plt.scatter(X[y_hc== 0, 0], X[y_hc == 0, 1], s = 100, c ="red", label="Cluster1")
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 100, c ="blue", label="Cluster2")
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 100, c ="green", label="Cluster3")
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s = 100, c ="cyan", label="Cluster4")
plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s = 100, c ="magenta", label="Cluster5")
plt.title("Cluster of clients")
plt.xlabel("Annual income (k$)")
plt.ylabel("Spending score (1-100)")
plt.legend()
plt.show()