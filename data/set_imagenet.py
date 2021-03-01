import os
import shutil
from PIL import Image

base = "./imagenet"
if not os.path.exists(base):
    os.makedirs(base)

class_info = "/data/ILSVRC2012_img_train/ILSVRC2012_classmap.txt")
f_labels = open(os.path.join(base, "labels.txt"), 'w')
f_names = open(os.path.join(base, "name.list"), 'w')
with open(class_info, 'r') as f:
    infos = f.readlines()
    for i in range(len(infos)):
        tmp = infos[i].split()
        f_labels.write(tmp[0] + '\n')
        f_names.write(tmp[2] + '\n')


print("To make train.list & val.list, use next 2 commands below")
print("""
        $ find /data/ILSVRC2012_img_train -name \*.JPEG > imagenet/train.list
        $ find /data/ILSVRC2012_img_val -name \*.JPEG > imagenet/val.list
        """)

