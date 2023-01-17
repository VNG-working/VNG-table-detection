import os, sys, shutil
from glob import glob

print(os.getcwd())

for i in range(2):
    os.chdir("..")

print(os.getcwd())

main_data_dir = os.getcwd() + "/data/table_detection/yolo_data"
print(os.listdir(main_data_dir))

imgs = glob(main_data_dir + "/images/*")
lbls = glob(main_data_dir + "/labels/*")

imgs.sort()
lbls.sort()

print("Imgs: {}".format(len(imgs)))
print("Lbls: {}".format(len(lbls)))

ratio = {
    "train" : 0.8,
    "valid" : 0.1,
    "test" : 0.1
}

train_index = int(len(imgs) * ratio["train"])
valid_index = int(len(imgs) * (ratio["train"] + ratio["valid"]))

print("train_index: {}".format(train_index))
print("valid_index: {}".format(valid_index))

train_imgs = imgs[:train_index]
valid_imgs = imgs[train_index:valid_index]
test_imgs = imgs[valid_index:]

train_lbls = lbls[:train_index]
valid_lbls = lbls[train_index:valid_index]
test_lbls = lbls[valid_index:]

print("train_imgs: {}".format(len(train_imgs)))
print("valid_imgs: {}".format(len(valid_imgs)))
print("test_imgs: {}".format(len(test_imgs)))
print("train_lbls: {}".format(len(train_lbls)))
print("valid_lbls: {}".format(len(valid_lbls)))
print("test_lbls: {}".format(len(test_lbls)))

save_data_dir = os.getcwd() + "/data/table_detection/yolo_yaml_data"

save_train_dir = save_data_dir + "/train"
save_test_dir = save_data_dir + "/test"
save_valid_dir = save_data_dir + "/valid"

for img_src, lbl_src in zip(test_imgs, test_lbls):
    imgs_name = img_src.split("/")[-1]
    lbls_name = lbl_src.split("/")[-1]

    img_dst = save_test_dir + "/images/{}".format(imgs_name)
    lbl_dst = save_test_dir + "/labels/{}".format(lbls_name)

    shutil.copy(img_src, img_dst)
    shutil.copy(lbl_src, lbl_dst)

for img_src, lbl_src in zip(valid_imgs, valid_lbls):
    imgs_name = img_src.split("/")[-1]
    lbls_name = lbl_src.split("/")[-1]

    img_dst = save_valid_dir + "/images/{}".format(imgs_name)
    lbl_dst = save_valid_dir + "/labels/{}".format(lbls_name)

    shutil.copy(img_src, img_dst)
    shutil.copy(lbl_src, lbl_dst)

for img_src, lbl_src in zip(train_imgs, train_lbls):
    imgs_name = img_src.split("/")[-1]
    lbls_name = lbl_src.split("/")[-1]

    img_dst = save_train_dir + "/images/{}".format(imgs_name)
    lbl_dst = save_train_dir + "/labels/{}".format(lbls_name)

    shutil.copy(img_src, img_dst)
    shutil.copy(lbl_src, lbl_dst)