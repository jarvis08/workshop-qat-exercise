
import os
import struct
import sys
import numpy

from array import array
from os import path
from PIL import Image #imported from pillow


def read_mnist(dataset):
    if dataset is "train":

        fname_img = "train-images.idx3-ubyte"
        fname_lbl = "train-labels.idx1-ubyte"

    elif dataset is "test":

        fname_img = "t10k-images.idx3-ubyte"
        fname_lbl = "t10k-labels.idx1-ubyte"

    else:
        raise ValueError("dataset must be 'test' or 'train'")

    flbl = open(fname_lbl, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = array("b", flbl.read())
    flbl.close()

    fimg = open(fname_img, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = array("B", fimg.read())
    fimg.close()

    return lbl, img, size, rows, cols


def into_cifar_format(labels, data, size, rows, cols, output_dir, itoa):
    # write data
    for (i, label) in enumerate(labels):
        output_filename = output_dir + '/' + str(i) + '_' + itoa[label] + ".jpg"
        print("writing " + output_filename)

        with open(output_filename, "wb") as h:
            data_i = [
                data[ (i*rows*cols + j*cols) : (i*rows*cols + (j+1)*cols) ]
                for j in range(rows)
            ]
            data_array = numpy.asarray(data_i)


            im = Image.fromarray(data_array)
            im.save(output_filename)


if __name__ == "__main__":
    mnist_dataset = ["train-images.idx3-ubyte",
                     "train-labels.idx1-ubyte",
                     "t10k-images.idx3-ubyte",
                     "t10k-labels.idx1-ubyte"]
    for file in mnist_dataset:
        if not path.exists(file):
            print("Locate ubyte files into 'data' directory.")
            sys.exit()

    if not path.exists('mnist'):
        os.makedirs('mnist')
        os.makedirs('mnist/test')
        os.makedirs('mnist/train')
    
    print("Write MNIST into CIFAR format..")
    str_label = {0:"zero", 1:"one", 2:"two", 3:"three", 4:"four", 5:"five", 6:"six", 7:"seven", 8:"eight", 9:"nine"}
    for dataset in ["train", "test"]:
        labels, data, size, rows, cols = read_mnist(dataset)
        into_cifar_format(labels, data, size, rows, cols,
                          path.join("mnist", dataset), str_label)

    print("Make label.list")
    with open("mnist/labels.txt", 'w') as f:
        for i in range(10):
            f.writelines(str_label[i] + "\n")

    print("Job Done.")

    print("To make train.list & test.list, use next 3 commands below")
    print("""
            $ cd mnist 
            $ find `pwd`/train -name \*.jpg > train.list
            $ find `pwd`/test -name \*.jpg > test.list
            """)

