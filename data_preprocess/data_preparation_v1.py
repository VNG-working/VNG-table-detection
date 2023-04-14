import os, shutil 
import random
from glob import glob
from xml.dom import minidom

print(os.getcwd())

for i in range(2):
    os.chdir("..")
print(os.getcwd())

def convert_coordinates(size, box):
    dw = 1.0/size[0]
    dh = 1.0/size[1]
    x = (box[0]+box[1])/2.0
    y = (box[2]+box[3])/2.0
    w = box[1]-box[0]
    h = box[3]-box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

lut={
    "table" : 0
}

def convert_xml2yolo( lut ):

    for fname in glob.glob("*.xml"):
        
        xmldoc = minidom.parse(fname)
        
        fname_out = (fname[:-4]+'.txt')

        with open(fname_out, "w") as f:

            itemlist = xmldoc.getElementsByTagName('object')
            size = xmldoc.getElementsByTagName('size')[0]
            width = int((size.getElementsByTagName('width')[0]).firstChild.data)
            height = int((size.getElementsByTagName('height')[0]).firstChild.data)

            for item in itemlist:
                # get class label
                classid =  (item.getElementsByTagName('name')[0]).firstChild.data
                if classid in lut:
                    label_str = str(lut[classid])
                else:
                    label_str = "-1"
                    print ("warning: label '%s' not in look-up table" % classid)

                # get bbox coordinates
                xmin = ((item.getElementsByTagName('bndbox')[0]).getElementsByTagName('xmin')[0]).firstChild.data
                ymin = ((item.getElementsByTagName('bndbox')[0]).getElementsByTagName('ymin')[0]).firstChild.data
                xmax = ((item.getElementsByTagName('bndbox')[0]).getElementsByTagName('xmax')[0]).firstChild.data
                ymax = ((item.getElementsByTagName('bndbox')[0]).getElementsByTagName('ymax')[0]).firstChild.data
                b = (float(xmin), float(xmax), float(ymin), float(ymax))
                bb = convert_coordinates((width,height), b)
                #print(bb)

                f.write(label_str + " " + " ".join([("%.6f" % a) for a in bb]) + '\n')

        print ("wrote %s" % fname_out)


data_dir = os.getcwd() + "/data/used_data"

img_paths = glob(data_dir + "/*.jpg")

def fullfill_data():
    for img_path in img_paths:
        filename = img_path.split("/")[-1][:-4]

        xml_path = data_dir + "/{}.xml".format(filename)
        if not os.path.exists(xml_path):
            txt_path = data_dir + "/{}.txt".format(filename)
            with open(txt_path, "w") as f:
                f.close()
        else:
            xmldoc = minidom.parse(xml_path)
            txt_path = data_dir + "/{}.txt".format(filename)
            with open(txt_path, "w") as f:
                itemlist = xmldoc.getElementsByTagName('object')
                size = xmldoc.getElementsByTagName('size')[0]
                width = int((size.getElementsByTagName('width')[0]).firstChild.data)
                height = int((size.getElementsByTagName('height')[0]).firstChild.data)

                for item in itemlist:
                    # get class label
                    classid =  (item.getElementsByTagName('name')[0]).firstChild.data
                    if classid in lut:
                        label_str = str(lut[classid])
                    else:
                        label_str = "-1"
                        print ("warning: label '%s' not in look-up table" % classid)

                    # get bbox coordinates
                    xmin = ((item.getElementsByTagName('bndbox')[0]).getElementsByTagName('xmin')[0]).firstChild.data
                    ymin = ((item.getElementsByTagName('bndbox')[0]).getElementsByTagName('ymin')[0]).firstChild.data
                    xmax = ((item.getElementsByTagName('bndbox')[0]).getElementsByTagName('xmax')[0]).firstChild.data
                    ymax = ((item.getElementsByTagName('bndbox')[0]).getElementsByTagName('ymax')[0]).firstChild.data
                    b = (float(xmin), float(xmax), float(ymin), float(ymax))
                    bb = convert_coordinates((width,height), b)
                    #print(bb)

                    f.write(label_str + " " + " ".join([("%.6f" % a) for a in bb]) + '\n')
            
            print ("wrote %s" % filename)

def copy_data():

    save_dir = os.getcwd() + "/data/table_detection/yolo_yaml_data/train"
    txt_paths = glob(data_dir + "/*.txt")

    for img_path, txt_path in zip(img_paths, txt_paths):

        img_name = img_path.split("/")[-1]
        img_dst = save_dir + "/images/{}".format(img_name)
        shutil.copy(img_path, img_dst)

        txt_name = txt_path.split("/")[-1]
        txt_dst = save_dir + "/labels/{}".format(img_name)
        shutil.copy(txt_path, txt_dst)

def verify():
    save_dir = os.getcwd() + "/data/table_detection/yolo_yaml_data/train"

    imgs = glob(save_dir + "/images/*")
    txt = glob(save_dir + "/labels/*")

    # print(len(imgs))
    # print(len(txt))

    # txt_paths = glob(data_dir + "/*.txt")

    # print(len(img_paths))
    # print(len(txt_paths))

    for img in imgs:
        if img.split("/")[-1][:-4] + ".txt" not in txt:
            print(img.split("/")[-1][:-4])

    
if __name__ == "__main__":
    # fullfill_data()

    # copy_data()

    verify()