import sys
import os
import random

random.seed(0)

DATA = sys.argv[1]
K = int(sys.argv[2])
n_train = 0
n_test = 0

if DATA == "cifar":
    n_train = 50000
    n_test = 10000
elif DATA == "svhn":
    n_train = 73257
    n_test = 26032

f_name = os.path.join(DATA, "train_random.txt")
with open(f_name, 'w') as f:
    for _ in range(n_train):
        f.write("{}\n".format(random.randint(0, K - 1)))

f_name = os.path.join(DATA, "test_random.txt")
with open(f_name, 'w') as f:
    for _ in range(n_test):
        f.write("{}\n".format(random.randint(0, K - 1)))
