import os
import numpy as np
import struct
import scipy.io as sio
import matplotlib.pyplot as plt

def save_in_darknet_form(data_path, save_path, itoa):
    mat = sio.loadmat(data_path)
    data = mat['X']
    y = mat['y']
    y = y.reshape(y.shape[0],)
    for i in range(data.shape[3]):
        plt.figure()
        plt.imsave(os.path.join(save_path, "{}_{}.png".format(i, itoa[y[i]])), data[..., i])
        plt.close()
        # break

if __name__ == "__main__":
    base = 'svhn'
    dataset_name = "_32x32.mat"
    dataset_type = ["train", "test"]

    train_path = base + '/train'
    test_path = base + '/test'
    if not os.path.exists(base):
        os.makedirs(base)
    if not os.path.exists(train_path):
        os.makedirs(train_path)
    if not os.path.exists(test_path):
        os.makedirs(test_path)
    
    print("Write SVHN into CIFAR format..")
    str_label = {1:"one", 2:"two", 3:"three", 4:"four", 5:"five", 6:"six", 7:"seven", 8:"eight", 9:"nine", 10:"zero"}
    for t in dataset_type:
        save_in_darknet_form(os.path.join(base, t + dataset_name), os.path.join(base, t), str_label)


    print("Make label.list")
    with open(os.path.join(base, "labels.txt"), 'w') as f:
        for i in range(1, 11):
            f.writelines(str_label[i] + "\n")

    print("Job Done.")

    print("To make train.list & test.list, use next 3 commands below")
    print("""
            $ cd svhn 
            $ find `pwd`/train -name \*.png > train.list
            $ find `pwd`/test -name \*.png > test.list
            """)

