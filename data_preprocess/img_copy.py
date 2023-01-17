from glob import glob
import os
import shutil

print(os.getcwd())

for i in range(2):
    os.chdir("..")

main_data_dir = os.getcwd() + "/data/table_detection"
print(os.listdir(main_data_dir))
data_corrected_dir = main_data_dir + "/data_corrected"
data_not_corrected_dir = main_data_dir + "/data_not_corrected"

corrected_imgs = glob(data_corrected_dir + "/*.jpg")
not_corrected_imgs = glob(data_not_corrected_dir + "/*.jpg")

print("img corrected: {}".format(len(corrected_imgs)))
print("img not corrected: {}".format(len(not_corrected_imgs)))

full_imgs = corrected_imgs + not_corrected_imgs

save_data = main_data_dir + "/yolo_data/images"

for img_path in full_imgs:
    filename = img_path.split("/")[-1]
    dst = save_data + f"/{filename}"
    shutil.copy(img_path, dst)
