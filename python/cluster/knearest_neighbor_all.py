import sys
import os

from sklearn.neighbors import NearestCentroid
import joblib
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

dataset_name = sys.argv[1]
npy_path = dataset_name + "/npy_dataset"


def load_npy(path):
    train = np.load("{}/train_data.npy".format(path))
    test = np.load("{}/test_data.npy".format(path))
    return train, test


def load_label(path):
    labels = []
    with open(os.path.join(path, 'labels.txt'), 'r') as f:
        lines = f.readlines()
        for l in lines:
            labels.append(l.replace('\n', ''))
    with open(os.path.join(path, 'train.list'), 'r') as f:
        lines = f.readlines()
        n_train = len(lines)
        y_train = np.zeros((n_train,), dtype=int)
        for i in range(n_train):
            for l in labels:
                if l in lines[i]:
                    y_train[i] = labels.index(l)
    return y_train


print("Load dataset")
train_data, test_data = load_npy(npy_path)
train_label = load_label(os.path.join("../../data", dataset_name))

print(train_data.shape)
print(test_data.shape)

print("Fit Model")
clf = NearestCentroid()
clf.fit(train_data, train_label)
print(clf.centroids_)

print("Save as txt file")
np.savetxt(dataset_name + '/train_nn_all.txt', train_label, delimiter=',', fmt='%i')
np.savetxt(dataset_name + '/test_nn_all.txt', clf.predict(test_data), delimiter=',', fmt='%i')

## Plot train dataset
#from sklearn.decomposition import PCA
#colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:brown', 'tab:cyan', 'tab:olive', 'tab:purple', 'tab:gray', 'tab:red', 'tab:pink']
#print("Plot train dataset..")
#y_kmeans = kmeans.predict(train_data)
#pca = PCA(n_components=2)
#pca.fit(train_data)
#train_pca = pca.transform(train_data)
#indices = []
#for i in range(K):
#    indices.append(np.where(y_kmeans == i))
#for i in range(K):
#    plt.scatter(train_pca[indices[i], 0], train_pca[indices[i], 1], c=colors[i], s=10, label=i, alpha=0.7, edgecolors='none')
#plt.legend()
#
#centers = kmeans.cluster_centers_
#centers = pca.transform(centers)
#for i in range(K):
#    plt.scatter(centers[i, 0], centers[i, 1], c=colors[i], s=30, label=i, alpha=0.7, edgecolors='black', linewidth=2)
#
#plt.suptitle('Train Dataset')
#plt.xlabel('Component 1')
#plt.ylabel('Component 2')
#plt.savefig(dataset_name + '/train_all.png', dpi=500)
#plt.clf()
#
## Plot test dataset
#print("Plot test dataset..")
#y_kmeans = kmeans.predict(test_data)
#test_pca = pca.transform(test_data)
#indices = []
#for i in range(K):
#    indices.append(np.where(y_kmeans == i))
#for i in range(K): plt.scatter(test_pca[indices[i], 0], test_pca[indices[i], 1], c=colors[i], s=10, label=i, alpha=0.7, edgecolors='none')
#plt.legend()
#
#centers = kmeans.cluster_centers_
#centers = pca.transform(centers)
#for i in range(K):
#    plt.scatter(centers[i, 0], centers[i, 1], c=colors[i], s=30, label=i, alpha=0.7, edgecolors='black', linewidth=2)
#plt.suptitle('Test Dataset')
#plt.xlabel('Component 1')
#plt.ylabel('Component 2')
#plt.savefig(dataset_name + '/test_all.png', dpi=500)
#plt.clf()
#
