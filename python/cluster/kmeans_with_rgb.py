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
K = 4


def load_npy(path):
    train = np.load("{}/train_data.npy".format(path))
    test = np.load("{}/test_data.npy".format(path))
    return train, test


print("Make dataset")
train_orig, test_orig = load_npy(npy_path)

train_data = 0
test_data = 0
for i in range(3):
    c = i * 1024
    train_min = np.min(train_orig[:, c:c+1024], axis=1)
    train_max = np.max(train_orig[:, c:c+1024], axis=1)
    test_min = np.min(test_orig[:, c:c+1024], axis=1)
    test_max = np.max(test_orig[:, c:c+1024], axis=1)

    train_min = train_min.reshape(train_min.shape[0], 1)
    train_max = train_max.reshape(train_max.shape[0], 1)
    train = np.append(train_min, train_max, axis=1)

    test_min = test_min.reshape(test_min.shape[0], 1)
    test_max = test_max.reshape(test_max.shape[0], 1)
    test = np.append(test_min, test_max, axis=1)
    if not i:
        train_data = np.copy(train)
        test_data = np.copy(test)
    else:
        train_data = np.append(train_data, train, axis=1)
        test_data = np.append(test_data, test, axis=1)

print(train_data.shape)
print(test_data.shape)

print("Fit Model")
kmeans = KMeans(n_clusters=K, random_state=0)
kmeans.fit(train_data)

print("Save model")
joblib.dump(kmeans, dataset_name + '/rgb.pkl') 

print("Save as txt file")
np.savetxt(dataset_name + '/train_rgb.txt', kmeans.labels_, delimiter=',', fmt='%i')
np.savetxt(dataset_name + '/test_rgb.txt', kmeans.predict(test_data), delimiter=',', fmt='%i')

from sklearn.decomposition import PCA
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:brown', 'tab:cyan', 'tab:olive', 'tab:purple', 'tab:gray', 'tab:red', 'tab:pink']
# Plot train dataset
print("Plot train dataset..")
y_kmeans = kmeans.predict(train_data)
pca = PCA(n_components=2)
pca.fit(train_data)
train_pca = pca.transform(train_data)
indices = []
for i in range(K):
    indices.append(np.where(y_kmeans == i))
for i in range(K):
    plt.scatter(train_pca[indices[i], 0], train_pca[indices[i], 1], c=colors[i], s=10, label=i, alpha=0.7, edgecolors='none')
plt.legend()

centers = kmeans.cluster_centers_
centers = pca.transform(centers)
for i in range(K):
    plt.scatter(centers[i, 0], centers[i, 1], c=colors[i], s=30, label=i, alpha=0.7, edgecolors='black', linewidth=2)

plt.suptitle('Train Dataset')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.savefig(dataset_name + '/train_rgb.png', dpi=500)
plt.clf()

# Plot test dataset
print("Plot test dataset..")
y_kmeans = kmeans.predict(test_data)
test_pca = pca.transform(test_data)
indices = []
for i in range(K):
    indices.append(np.where(y_kmeans == i))
for i in range(K): plt.scatter(test_pca[indices[i], 0], test_pca[indices[i], 1], c=colors[i], s=10, label=i, alpha=0.7, edgecolors='none')
plt.legend()

centers = kmeans.cluster_centers_
centers = pca.transform(centers)
for i in range(K):
    plt.scatter(centers[i, 0], centers[i, 1], c=colors[i], s=30, label=i, alpha=0.7, edgecolors='black', linewidth=2)
plt.suptitle('Test Dataset')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.savefig(dataset_name + '/test_rgb.png', dpi=500)
plt.clf()

# Plot train dataset
y_kmeans = kmeans.predict(train_data)
indices = []
for i in range(K):
    indices.append(np.where(y_kmeans == i))
for ch in range(3):
    plt.clf()
    f_name = ''
    if not ch:
        f_name = dataset_name + '/train_rgb_r.png'
        print("Plot train dataset's R-channel")
    elif ch == 1:
        f_name = dataset_name + '/train_rgb_g.png'
        print("Plot train dataset's G-channel")
    else:
        f_name = dataset_name + '/train_rgb_b.png'
        print("Plot train dataset's B-channel")

    for i in range(K):
        plt.scatter(train_data[indices[i], ch * 2], train_data[indices[i], ch * 2 + 1], c=colors[i], s=10, label=i, alpha=0.7, edgecolors='none')
    plt.legend()

    centers = kmeans.cluster_centers_
    for i in range(K):
        plt.scatter(centers[i, ch * 2], centers[i, ch * 2 + 1], c=colors[i], s=30, label=i, alpha=0.7, edgecolors='black', linewidth=2)
    plt.suptitle('Train Dataset')
    plt.xlabel('Min Value')
    plt.ylabel('Max Value')
    plt.savefig(f_name, dpi=500)
    plt.clf()

# Plot test dataset
y_kmeans = kmeans.predict(test_data)
indices = []
for i in range(K):
    indices.append(np.where(y_kmeans == i))
for ch in range(3):
    plt.clf()
    f_name = ''
    if not ch:
        f_name = dataset_name + '/test_rgb_r.png'
        print("Plot train dataset's R-channel")
    elif ch == 1:
        f_name = dataset_name + '/test_rgb_g.png'
        print("Plot train dataset's G-channel")
    else:
        f_name = dataset_name + '/test_rgb_b.png'
        print("Plot train dataset's B-channel")
    for i in range(K):
        plt.scatter(test_data[indices[i], ch * 2], test_data[indices[i], ch * 2 + 1], c=colors[i], s=10, label=i, alpha=0.7, edgecolors='none')
    plt.legend()

    centers = kmeans.cluster_centers_
    for i in range(K):
        plt.scatter(centers[i, ch * 2], centers[i, ch * 2 + 1], c=colors[i], s=30, label=i, alpha=0.7, edgecolors='black', linewidth=2)
    plt.suptitle('Test Dataset')
    plt.xlabel('Min Value')
    plt.ylabel('Max Value')
    plt.savefig(f_name, dpi=500)
    plt.clf()

