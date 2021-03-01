import sys

from sklearn.cluster import KMeans
import numpy as np
import joblib

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

dataset_name = sys.argv[1]
npy_path = dataset_name + "/npy_dataset"

#K = 10
K = 3
#K = 4
N_PARTITION = 4
#N_PARTITION = 9

# Only used for images size of 32x32x3
n_el = int(np.sqrt(1024 / N_PARTITION))
print(n_el)


def load_npy(path):
    train = np.load("{}/train_data.npy".format(path))
    test = np.load("{}/test_data.npy".format(path))
    return train, test


print("Make dataset")
train_orig, test_orig = load_npy(npy_path)

train_data = np.array([])
test_data = np.array([])
start = 0
for i in range(3):
    c = i * 1024
    for j in range(N_PARTITION):
        if j < N_PARTITION / 2:
            start = c + (j * n_el)
        else:
            start = c + 512 + (j - 2)  * n_el
        tr = np.array([])
        te = np.array([])
        for k in range(n_el):
            end = start + n_el
            if not k:
                tr = np.copy(train_orig[:, start:end])
                te = np.copy(test_orig[:, start:end])
            else:
                tr = np.concatenate((tr, train_orig[:, start:end]), axis=1)
                te = np.concatenate((te, test_orig[:, start:end]), axis=1)
            start += 2 * n_el

        tr_min = np.min(tr, axis=1)
        tr_max = np.max(tr, axis=1)
        te_min = np.min(te, axis=1)
        te_max = np.max(te, axis=1)

        tr_min = tr_min.reshape(tr_min.shape[0], 1)
        tr_max = tr_max.reshape(tr_max.shape[0], 1)
        te_min = te_min.reshape(te_min.shape[0], 1)
        te_max = te_max.reshape(te_max.shape[0], 1)

        train = np.append(tr_min, tr_max, axis=1)
        test = np.append(te_min, te_max, axis=1)

        if not i and not j:
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
joblib.dump(kmeans, dataset_name + '/partition.pkl') 

print("Save as txt file")
np.savetxt(dataset_name + '/train_partition.txt', kmeans.labels_, delimiter=',', fmt='%i')
np.savetxt(dataset_name + '/test_partition.txt', kmeans.predict(test_data), delimiter=',', fmt='%i')

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
plt.savefig(dataset_name + '/train_partition.png', dpi=500)
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
plt.savefig(dataset_name + '/test_partition.png', dpi=500)
plt.clf()

# Plot train dataset
y_kmeans = kmeans.predict(train_data)
indices = []
for i in range(K):
    indices.append(np.where(y_kmeans == i))
for p in range(N_PARTITION):
    plt.clf()
    f_name = dataset_name + '/train_partition_{}.png'.format(p)
    print("Plot train dataset's partition-{}".format(p))

    for i in range(K):
        plt.scatter(train_data[indices[i], p * 2], train_data[indices[i], p * 2 + 1], c=colors[i], s=10, label=i, alpha=0.7, edgecolors='none')
    plt.legend()

    centers = kmeans.cluster_centers_
    for i in range(K):
        plt.scatter(centers[i, p * 2], centers[i, p * 2 + 1], c=colors[i], s=30, label=i, alpha=0.7, edgecolors='black', linewidth=2)
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
for p in range(N_PARTITION):
    plt.clf()
    f_name = dataset_name + '/test_partition_{}.png'.format(p)
    print("Plot test dataset's partition-{}".format(p))

    for i in range(K):
        plt.scatter(test_data[indices[i], p * 2], test_data[indices[i], p * 2 + 1], c=colors[i], s=10, label=i, alpha=0.7, edgecolors='none')
    plt.legend()

    centers = kmeans.cluster_centers_
    for i in range(K):
        plt.scatter(centers[i, p * 2], centers[i, p * 2 + 1], c=colors[i], s=30, label=i, alpha=0.7, edgecolors='black', linewidth=2)
    plt.suptitle('Test Dataset')
    plt.xlabel('Min Value')
    plt.ylabel('Max Value')
    plt.savefig(f_name, dpi=500)
    plt.clf()

