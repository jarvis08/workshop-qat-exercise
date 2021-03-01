import os
import numpy as np
from PIL import Image
import sys

dataset_name = sys.argv[1]
if not os.path.exists(dataset_name):
    os.mkdir(dataset_name)

save_path = dataset_name + "/npy_dataset"
if not os.path.exists(save_path):
    os.mkdir(save_path)


def imagelist_extract(list_name):
    image_list = np.array([])

    for f in list_name:
        im = np.array(Image.open(f))
        im = np.concatenate((im[:, :, 0].reshape(-1),
                             im[:, :, 1].reshape(-1),
                             im[:, :, 2].reshape(-1))) / 255
        im = np.expand_dims(im, axis=0)
        if len(image_list) != 0:
            image_list = np.append(image_list, im, axis=0)
        else:
            image_list = np.copy(im)
    return image_list


print(">> Load train images..")
path = '../../data/{}/train.list'.format(dataset_name)
with open(path, 'r') as f:
    train_list = f.readlines()
train_list = [line.split('\n')[0] for line in train_list]
train_data = imagelist_extract(train_list)

print("Save to train_data.npy")
with open(save_path + "/train_data.npy", "wb") as f:
    np.save(f, train_data)
print("Done.")


print(">> Load test images..")
path = '../../data/{}/test.list'.format(dataset_name)
with open(path, 'r') as f:
    test_list = f.readlines()
test_list = [line.split('\n')[0] for line in test_list]
test_data = imagelist_extract(test_list)

print("Save to test_data.npy")
with open(save_path + "/test_data.npy", "wb") as f:
    np.save(f, test_data)
print("Done.")
