import os
import numpy as np
from PIL import Image
import sys
import time

DATASET = "tiny-imagenet-200"
if not os.path.exists(DATASET):
    os.mkdir(DATASET)

save_path = os.path.join(DATASET, "npy_dataset")
if not os.path.exists(save_path):
    os.mkdir(save_path)


def get_minmax_from_image(path):
    im = np.array(Image.open(path))
    im = im.reshape(-1) / 255
    return np.array([[np.min(im), np.max(im)]])


print(">> Load train images..")
path = '../../data/{}/train.list'.format(DATASET)
with open(path, 'r') as f:
    train_list = f.readlines()
    train_list = [line.split('\n')[0] for line in train_list]
    n_train = len(train_list)

minmax = get_minmax_from_image(train_list[0])
start = time.time()
for i in range(1, n_train):
    print("Train {}/{}".format(i, n_train))
    minmax = np.append(minmax, get_minmax_from_image(train_list[i]), axis=0)
print("Done, {} sec.".format(time.time() - start))
path = os.path.join(save_path, "train_minmax.npy")
print("Make {}".format(path))
with open(path, "wb") as f:
    np.save(f, minmax)
del minmax


print(">> Load test images..")
path = '../../data/{}/val.list'.format(DATASET)
with open(path, 'r') as f:
    test_list = f.readlines()
    test_list = [line.split('\n')[0] for line in test_list]
    n_test = len(test_list)
minmax = get_minmax_from_image(test_list[0])
start = time.time()
for i in range(1, n_test):
    print("Test {}/{}".format(i, n_test))
    minmax = np.append(minmax, get_minmax_from_image(test_list[i]), axis=0)
print("Done, {} sec.".format(time.time() - start))

path = os.path.join(save_path, "test_minmax.npy")
print("Make {}".format(path))
with open(path, "wb") as f:
    np.save(f, minmax)
