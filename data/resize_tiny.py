from PIL import Image

print(">> Load train images..")
with open('./tiny-imagenet-200/train.list', 'r') as f:
    train_list = f.readlines()

print(">> Load val images..")
with open('./tiny-imagenet-200/val.list', 'r') as f:
    valid_list = f.readlines()


print(">> Resize train dataset\n")
for d in train_list:
    path = d.split("\n")[0]
    img = Image.open(path)
    img_resize = img.resize((224, 224))
    img_resize.save(path)


print(">> Resize valid dataset\n")
for d in valid_list:
    path = d.split("\n")[0]
    img = Image.open(path)
    img_resize = img.resize((224, 224))
    img_resize.save(path)
