# -*- coding: utf-8 -*-

from xml.dom import minidom
import os
import glob

print(os.getcwd())

for i in range(2):
    os.chdir("..")

main_data_dir = os.getcwd() + "/data/table_detection"
print(os.listdir(main_data_dir))
data_corrected_dir = main_data_dir + "/data_corrected"
data_not_corrected_dir = main_data_dir + "/data_not_corrected"

data_corrected_lst = glob.glob(data_corrected_dir + "/*.xml")
data_not_corrected_lst = glob.glob(data_not_corrected_dir + "/*.xml")
print("data corrected: {}".format(len(data_corrected_lst)))
print("data not corrected: {}".format(len(data_not_corrected_lst)))

full_xmls = data_corrected_lst + data_not_corrected_lst

save_data = main_data_dir + "/yolo_data/labels"

lut={}
lut["table"] = 0

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

def convert_xml2yolo(lut, lst = full_xmls):

    for fname in full_xmls:
        
        xmldoc = minidom.parse(fname)
        
        filename = fname.split("/")[-1]
        
        fname_out = save_data + "/" + filename[:-4] + '.txt'

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

def main():
    convert_xml2yolo( lut )


if __name__ == '__main__':
    main()