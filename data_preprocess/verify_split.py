import os
from glob import glob

for i in range(2):
    os.chdir("..")

main_data_dir = os.getcwd() + "/data/table_detection/yolo_yaml_data"

test_dir = main_data_dir + "/test"
train_dir = main_data_dir + "/train"
valid_dir = main_data_dir + "/valid"

test_imgs = glob(test_dir + "/images/*")
test_lbls = glob(test_dir + "/labels/*")

test_imgs.sort()
test_lbls.sort()

train_imgs = glob(train_dir + "/images/*")
train_lbls = glob(train_dir + "/labels/*")

train_imgs.sort()
train_lbls.sort()

valid_imgs = glob(valid_dir + "/images/*")
valid_lbls = glob(valid_dir + "/labels/*")

valid_imgs.sort()
valid_lbls.sort()

# count

print("Test: {0} - {1}".format(len(test_imgs), len(test_lbls)))
print("Train: {0} - {1}".format(len(train_imgs), len(train_lbls)))
print("Valid: {0} - {1}".format(len(valid_imgs), len(valid_lbls)))

# same file

test_wrong, train_wrong, valid_wrong = {"imgs" : [], "lbls" : []}, \
                                        {"imgs" : [], "lbls" : []}, \
                                        {"imgs" : [], "lbls" : []}
for x, y in zip(test_imgs, test_lbls):
    img_name = x.split("/")[-1].split(".")[0]
    lbl_name = y.split("/")[-1].split(".")[0]

    if img_name != lbl_name:
        test_wrong["imgs"].append(img_name)
        test_wrong["lbls"].append(lbl_name)

for x, y in zip(train_imgs, train_lbls):
    img_name = x.split("/")[-1].split(".")[0]
    lbl_name = y.split("/")[-1].split(".")[0]

    if img_name != lbl_name:
        train_wrong["imgs"].append(img_name)
        train_wrong["lbls"].append(lbl_name)

for x, y in zip(valid_imgs, valid_lbls):
    img_name = x.split("/")[-1].split(".")[0]
    lbl_name = y.split("/")[-1].split(".")[0]

    if img_name != lbl_name:
        valid_wrong["imgs"].append(img_name)
        valid_wrong["lbls"].append(lbl_name)

print(test_wrong)
print(train_wrong)
print(valid_wrong)