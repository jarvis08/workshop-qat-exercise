import os
import numpy as np
from PIL import Image
import sys
import time

dataset_name = "tiny-imagenet-200"
if not os.path.exists(dataset_name):
    os.mkdir(dataset_name)

save_path = dataset_name + "/npy_dataset"
if not os.path.exists(save_path):
    os.mkdir(save_path)


def get_mean_from_image(path):
    im = np.array(Image.open(path))
    im = im.reshape(-1) / 255
    return np.array([[np.mean(im)]])


print(">> Load train images..")
path = '../../data/{}/train.list'.format(dataset_name)
with open(path, 'r') as f:
    train_list = f.readlines()
    train_list = [line.split('\n')[0] for line in train_list]
    n_train = len(train_list)

means = get_mean_from_image(train_list[0])
start = time.time()
for i in range(1, n_train):
    print("Train {}/{}".format(i, n_train))
    means = np.append(means, get_mean_from_image(train_list[i]), axis=0)
print("Done, {} sec.".format(time.time() - start))
path = os.path.join(save_path, "train_means.npy")
print("Make {}".format(path))
with open(path, "wb") as f:
    np.save(f, means)
del means


print(">> Load test images..")
path = '../../data/{}/val.list'.format(dataset_name)
with open(path, 'r') as f:
    test_list = f.readlines()
    test_list = [line.split('\n')[0] for line in test_list]
    n_test = len(test_list)
means = get_mean_from_image(test_list[0])
start = time.time()
for i in range(1, n_test):
    print("Test {}/{}".format(i, n_test))
    means = np.append(means, get_mean_from_image(test_list[i]), axis=0)
print("Done, {} sec.".format(time.time() - start))

path = os.path.join(save_path, "test_means.npy")
print("Make {}".format(path))
with open(path, "wb") as f:
    np.save(f, means)
