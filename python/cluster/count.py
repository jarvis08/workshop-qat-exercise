from collections import Counter


with open("train_cluster.txt") as f:
    tmp = f.readlines()
    for i in range(len(tmp)):
        tmp[i] = tmp[i].replace('\n', '')
    print(Counter(tmp))


with open("test_cluster.txt") as f:
    tmp = f.readlines()
    for i in range(len(tmp)):
        tmp[i] = tmp[i].replace('\n', '')
    print(Counter(tmp))
