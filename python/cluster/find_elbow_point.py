import sys

from sklearn.cluster import KMeans
import numpy as np
import joblib

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

dataset_name = sys.argv[1]
npy_path = dataset_name + "/npy_dataset"

mode = int(sys.argv[2])
if not mode:
    mode = "minmax"
elif mode == 1:
    mode = "rgb"
elif mode == 2:
    if len(sys.argv) < 4:
        print("To use Partitioning, plz give the number of partitions as argument.")
        exit()
    mode = "partition"

BATCH_SIZE = 200
EPOCH_SIZE = 1


def rule_of_thumb(n_data):
    return np.sqrt(n_data)


def load_npy(path):
    train = np.load("{}/train_data.npy".format(path))
    test = np.load("{}/test_data.npy".format(path))
    return train, test


train_orig, test_orig = load_npy(npy_path)
k_values = list(range(1, 21))
print(k_values)
print("Number of Ks to try = {}".format(len(k_values)))

print("Load dataset")
train_orig, test_orig = load_npy(npy_path)

train_data = np.array([])
test_data = np.array([])

# Min/Max per Image
if mode == "minmax":
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

elif mode == "rgb":
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
else:
    N_PARTITION = int(sys.argv[3])
    n_el = int(np.sqrt(1024 / N_PARTITION))
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
            for k in range(N_PARTITION):
                start += (k * 2) * n_el
                end = start + n_el
                if not k:
                    tr = np.copy(train_orig[:, start:end])
                    te = np.copy(test_orig[:, start:end])
                else:
                    tr = np.concatenate((tr, train_orig[:, start:end]), axis=1)
                    te = np.concatenate((te, test_orig[:, start:end]), axis=1)

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

inertias = []
for k in k_values:
    print("Train {}-means..".format(k))
    model = KMeans(n_clusters=k, random_state=0)
    model.fit(train_data)
    inertias.append(model.inertia_)
plt.plot(k_values, inertias, marker='o')
plt.suptitle('K-means with {} Method'.format(mode.upper()))
plt.xlabel('K Value')
plt.ylabel('Inertia')
plt.savefig(dataset_name + '/inertia_{}.png'.format(mode), dpi=500)
plt.clf()

