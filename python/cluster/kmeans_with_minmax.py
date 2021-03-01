import sys

from sklearn.cluster import KMeans
import numpy as np
import joblib

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

dataset_name = sys.argv[1]
npy_path = dataset_name + "/npy_dataset"

BATCH_SIZE = 200
EPOCH_SIZE = 1
K = int(sys.argv[2])


def load_npy(path):
    train = np.load("{}/train_data.npy".format(path))
    test = np.load("{}/test_data.npy".format(path))
    return train, test


print("Make dataset")
train_orig, test_orig = load_npy(npy_path)

train_min = np.min(train_orig, axis=1)
train_max = np.max(train_orig, axis=1)

test_min = np.min(test_orig, axis=1)
test_max = np.max(test_orig, axis=1)

train_min = train_min.reshape(train_min.shape[0], 1)
train_max = train_max.reshape(train_min.shape[0], 1)
train_data = np.append(train_min, train_max, axis=1)

test_min = test_min.reshape(test_min.shape[0], 1)
test_max = test_max.reshape(test_min.shape[0], 1)
test_data = np.append(test_min, test_max, axis=1)

print(train_data.shape)
print(test_data.shape)

print("Fit Model")
kmeans = KMeans(n_clusters=K, random_state=0)
kmeans.fit(train_data)

print("Save model")
joblib.dump(kmeans, dataset_name + '/minmax.pkl') 

print("Save as txt file")
np.savetxt(dataset_name + '/train_minmax.txt', kmeans.labels_, delimiter=',', fmt='%i')
np.savetxt(dataset_name + '/test_minmax.txt', kmeans.predict(test_data), delimiter=',', fmt='%i')
print(kmeans.cluster_centers_)

colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:brown', 'tab:cyan', 'tab:olive', 'tab:purple', 'tab:gray', 'tab:red', 'tab:pink']

# Plot train dataset
print("Plot train dataset..")
y_kmeans = kmeans.predict(train_data)
indices = []
for i in range(K):
    indices.append(np.where(y_kmeans == i))
for i in range(K):
    plt.scatter(train_data[indices[i], 0], train_data[indices[i], 1], c=colors[i], s=10, label=i, alpha=0.7, edgecolors='none')
plt.legend()

centers = kmeans.cluster_centers_
for i in range(K):
    plt.scatter(centers[i, 0], centers[i, 1], c=colors[i], s=30, label=i, alpha=0.7, edgecolors='black', linewidth=2)

plt.suptitle('Train Dataset')
plt.xlabel('Min Value')
plt.ylabel('Max Value')
plt.savefig(dataset_name + '/train_minmax.png', dpi=500)
plt.clf()

# Plot test dataset
print("Plot test dataset..")
y_kmeans = kmeans.predict(test_data)
indices = []
for i in range(K):
    indices.append(np.where(y_kmeans == i))
for i in range(K):
    plt.scatter(test_data[indices[i], 0], test_data[indices[i], 1], c=colors[i], s=10, label=i, alpha=0.7, edgecolors='none')
plt.legend()

centers = kmeans.cluster_centers_
for i in range(K):
    plt.scatter(centers[i, 0], centers[i, 1], c=colors[i], s=30, label=i, alpha=0.7, edgecolors='black', linewidth=2)
plt.suptitle('Test Dataset')
plt.xlabel('Min Value')
plt.ylabel('Max Value')
plt.savefig(dataset_name + '/test_minmax.png', dpi=500)
