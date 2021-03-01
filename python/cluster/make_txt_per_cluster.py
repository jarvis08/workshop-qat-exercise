import os
import numpy as np
from PIL import Image
import sys

if len(sys.argv) < 4:
    ERROR = """You must give,
            1. Dataset name
            2. txt file of target training dataset's cluster info
            3. Number of clusters"""
    print(ERROR)
    exit()
dataset_name = sys.argv[1]
ctr_method = sys.argv[2].split('.')[0]
orig_path = os.path.join(dataset_name, sys.argv[2]) # e.g., train_minmax/rgb/partition.txt
N_CLUSTER = int(sys.argv[3])

orig_data = np.loadtxt(orig_path, dtype=np.int32)
for ctr in range(N_CLUSTER):
    save_path = os.path.join(dataset_name, "{}.c{}.txt".format(ctr_method, ctr))
    print("Save to {}".format(save_path))
    with open(save_path, 'w') as f:
        ctr_indices = list(np.where(orig_data == ctr))[0]
        ctr_indices = ctr_indices.tolist()
        ctr_indices = list(map(str, ctr_indices))
        f.write(str(len(ctr_indices)) + "\n")
        for i in range(len(ctr_indices)):
            ctr_indices[i] = ctr_indices[i] + "\n"
        f.writelines(ctr_indices)
