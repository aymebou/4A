## kmeans

from copy import deepcopy
from random import randint, choices, random
import numpy as np
from Clusterer import Clusterer
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_samples, silhouette_score

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


clusterer = Clusterer()

def sample(classNumber=3, size=50, dimension=2):
    centers = np.array([[randint(-classNumber,classNumber) for i in range(dimension)] for i in range(classNumber)])
    print("Centers used for generation:", centers, sep="\n")
    dataset = []
    for center in centers:
        for i in range(size):
            dataset.append(center + np.array([random()-0.5 for i in range(dimension)]))
    return (dataset)

dataset = np.array(sample())
print("SAMPLE DATA GENERATED: ")
print("- Dimension", len(dataset[0]))
print("- Number of clusters", 3)
print("- Sample size", len(dataset))

print("\n---  Testing k-means standard algorithm  ---")
try:
    categories, centroids, iterations = clusterer.kMeansFull(dataset, 3)
    print(f"Terminated with {iterations} iterations, found centroids:", clusterer.computeCentroids(categories), sep="\n")
except:
    print("Algorithm failed")

print("\n---  Testing k-means ++ algorithm  ---")
try:
    categories, centroids, iterations = clusterer.kMeansPlusPlus(dataset, 3)
    print(f"Terminated with {iterations} iterations, found centroids:", clusterer.computeCentroids(categories), sep="\n")
except:
    print("Algorithm failed")



###
###     PLOTTING SILOUHETTE
###

classes_to_test = [2, 3, 4, 5, 6]

for classNumber in classes_to_test:
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)
    ax1.set_xlim([-0.1, 1])
    ax1.set_ylim([0, len(dataset) + (classNumber + 1) * 10])
    cluster_labels, centroids = clusterer.predictLabels(dataset, classNumber)
    silhouette_avg = silhouette_score(dataset, cluster_labels)
    sample_silhouette_values = silhouette_samples(dataset, cluster_labels)
    y_lower = 10
    for i in range(classNumber):
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / classNumber)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        y_lower = y_upper + 10

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    colors = cm.nipy_spectral(cluster_labels.astype(float) / classNumber)
    ax2.scatter(dataset[:, 0], dataset[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                c=colors, edgecolor='k')

    ax2.scatter(centroids[:, 0], centroids[:, 1], marker='o',
                c="white", alpha=1, s=200, edgecolor='k')

    for i, c in enumerate(centroids):
        ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                    s=50, edgecolor='k')

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(("Silhouette analysis for KMeans clustering oKn sample data "
                  "with classNumber = %d" % classNumber),
                 fontsize=14, fontweight='bold')

plt.show()
