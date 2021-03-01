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

N_PARTITION = 4
n_el = int(np.sqrt(1024 / N_PARTITION)) # Only used for images size of 32x32x3


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
train_orig, test_orig = load_npy(npy_path)
train_label = load_label(os.path.join("../../data", dataset_name))

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
clf = NearestCentroid()
clf.fit(train_data, train_label)
np.savetxt(dataset_name + '/nn_partition_centroids.txt', clf.centroids_, delimiter=',', fmt='%f')

print("Save as txt file")
np.savetxt(dataset_name + '/train_nn_partition.txt', train_label, delimiter=',', fmt='%i')
np.savetxt(dataset_name + '/test_nn_partition.txt', clf.predict(test_data), delimiter=',', fmt='%i')
