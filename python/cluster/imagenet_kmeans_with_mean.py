import sys
import os

from sklearn.cluster import KMeans
import numpy as np
import joblib

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

DATASET = "tiny-imagenet-200"
npy_path = os.path.join(DATASET, "npy_dataset")

BATCH_SIZE = 200
EPOCH_SIZE = 1
K = 10


def load_npy(path):
    train = np.load(os.path.join(path, "train_means.npy"))
    test = np.load(os.path.join(path, "test_means.npy"))
    return train, test


print("Make dataset")
train_data, test_data = load_npy(npy_path)

print(train_data.shape)
print(test_data.shape)

print("Fit Model")
kmeans = KMeans(n_clusters=K, random_state=0)
kmeans.fit(train_data)

print("Save model")
joblib.dump(kmeans, os.path.join(DATASET, 'means.pkl')) 

print("Save as txt file")
np.savetxt(os.path.join(DATASET, 'train_means.txt'), kmeans.labels_, delimiter=',', fmt='%i')
np.savetxt(os.path.join(DATASET, 'test_means.txt'), kmeans.predict(test_data), delimiter=',', fmt='%i')
print(kmeans.cluster_centers_)

colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:brown', 'tab:cyan', 'tab:olive', 'tab:purple', 'tab:gray', 'tab:red', 'tab:pink']

# Plot train dataset
print("Plot train dataset..")
y_kmeans = kmeans.predict(train_data)
indices = []
for i in range(10):
    indices.append(np.where(y_kmeans == i))
for i in range(10):
    plt.scatter(train_data[indices[i]], train_data[indices[i]], c=colors[i], s=10, label=i, alpha=0.7, edgecolors='none')
plt.legend()

centers = kmeans.cluster_centers_
for i in range(10):
    plt.scatter(centers[i], centers[i], c=colors[i], s=30, label=i, alpha=0.7, edgecolors='black', linewidth=2)

plt.suptitle('Train Dataset')
plt.xlabel('Mean')
plt.ylabel('Mean')
plt.savefig(os.path.join(DATASET, 'train_means.png'), dpi=500)
plt.clf()

# Plot test dataset
print("Plot test dataset..")
y_kmeans = kmeans.predict(test_data)
indices = []
for i in range(10):
    indices.append(np.where(y_kmeans == i))
for i in range(10):
    plt.scatter(test_data[indices[i]], test_data[indices[i]], c=colors[i], s=10, label=i, alpha=0.7, edgecolors='none')
plt.legend()

centers = kmeans.cluster_centers_
for i in range(10):
    plt.scatter(centers[i], centers[i], c=colors[i], s=30, label=i, alpha=0.7, edgecolors='black', linewidth=2)
plt.suptitle('Test Dataset')
plt.xlabel('Mean')
plt.ylabel('Mean')
plt.savefig(os.path.join(DATASET, 'test_means.png'), dpi=500)
