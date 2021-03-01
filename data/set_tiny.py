import os
import shutil
from PIL import Image

base = "./tiny-imagenet-200"
train = base + "/train"
val = base + "/val"
classes = os.listdir(train)

# Training Dataset
for c in classes:
    per_class = train + "/" + c
    image_dir = per_class + "/images"
    images = os.listdir(image_dir)
    for i in images:
        os.rename(image_dir + "/" + i, train + "/" + i)
    shutil.rmtree(per_class)

# validation Dataset
with open(val + "/val_annotations.txt") as f:
    annotations = dict()
    tmp = f.readlines()
    for i in range(len(tmp)):
        t = tmp[i].split("\t")
        annotations[t[0]] = t[1]

image_dir = val + "/images"
images = os.listdir(image_dir)
for i in images:
    new = i.replace("val", annotations[i])
    os.rename(image_dir + "/" + i, val + "/" + new)
shutil.rmtree(image_dir)
os.remove(val + "/val_annotations.txt")

# make names file from wnids.txt
with open(os.path.join(base, "wnids.txt"), 'r') as f:
    wnids = f.readlines()
    for i in range(len(wnids)):
        wnids[i] = wnids[i].replace("\n", '')

names = []
with open(os.path.join(base, "words.txt"), 'r') as f:
    name_dict = {}
    words = f.readlines()
    for i in range(len(words)):
        words[i] = words[i].split("\t")
        words[i][1] = words[i][1].replace("\n", '')
        words[i][1] = words[i][1].split(', ')[0]
        name_dict[words[i][0]] = words[i][1]
print(name_dict)

with open(os.path.join(base, "name.list"), 'w') as f:
    for label in wnids:
        f.write(name_dict[label] + "\n")

print("To make train.list & val.list, use next 3 commands below")
print("""
        $ cd tiny-imagenet-200 
        $ find `pwd`/train -name \*.JPEG > train.list
        $ find `pwd`/val -name \*.JPEG > val.list
        """)

