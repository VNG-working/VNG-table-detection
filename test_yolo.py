import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import xml.etree.ElementTree as ET
from PIL import Image
from time import time
import logging
logging.basicConfig(level=logging.ERROR)

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, r, (dw, dh)

def scale_boxes(img1_shape, boxes, img0_shape):
    # Rescale boxes (xyxy) from img1_shape to img0_shape
    gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
    pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding

    boxes[..., [0, 2]] -= pad[0]  # x padding
    boxes[..., [1, 3]] -= pad[1]  # y padding
    boxes[..., :4] /= gain
    # clip_boxes(boxes, img0_shape)
    return boxes

def draw_boxes(src_image, boxes, scores, labels):
    color = (255, 0, 0)
    for box, score, lb in zip(boxes, scores, labels):
        xmin, ymin, xmax, ymax = box.astype('int32')
        score = '{:.4f}'.format(score)
        label = '-'.join([lb, score])
        ret, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(src_image, (xmin, ymin), (xmax, ymax), color, 1)
        cv2.rectangle(src_image, (xmin, ymax - ret[1] - baseline), (xmin + ret[0], ymax), color, -1)
        cv2.putText(src_image, label, (xmin, ymax - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    return src_image

def show(name, img):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, img)
    key = cv2.waitKey(0)
    return key

def to_xml(xml_path, imgname, boxes, labels, size):
    w, h = size
    root = ET.Element('annotations')
    filename = ET.SubElement(root, 'filename')
    filename.text = imgname
    size = ET.SubElement(root, 'size')
    width = ET.SubElement(size, 'width')
    height = ET.SubElement(size, 'height')
    width.text, height.text = str(w), str(h)
    depth = ET.SubElement(size, 'depth')
    depth.text = '3'
    for box, label in zip(boxes, labels):
        obj = ET.SubElement(root, 'object')
        name = ET.SubElement(obj, 'name')
        name.text = label
        bndbox = ET.SubElement(obj, 'bndbox')
        xmin, ymin = ET.SubElement(bndbox, 'xmin'), ET.SubElement(bndbox, 'ymin')
        xmax, ymax = ET.SubElement(bndbox, 'xmax'), ET.SubElement(bndbox, 'ymax')
        xmin.text, ymin.text, xmax.text, ymax.text = [str(b) for b in box]
    ET.ElementTree(root).write(xml_path)

class Detector:
    def __init__(self):
        
        model_path = os.path.join(os.getcwd(), "VNG-table-detection/best_saved_model")
        print(os.path.exists(model_path))
        self.model = tf.saved_model.load(model_path)
        self.labels = ['table']

    def preprocess(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image, ratio, dwdh = letterbox(image, auto=False)
        # image = image.transpose((2, 0, 1))
        # image = np.ascontiguousarray(image)

        im = image.astype(np.float32) / 255.0
        tensor_in = tf.expand_dims(im, 0)
        return tensor_in, ratio, dwdh

    def __call__(self, img, threshold=0.5):
        h, w = img.shape[:2]
        tensor, ratio, dwdh = self.preprocess(img)
        s = time()
        with tf.device("/CPU:0"):
            outs = self.model.signatures['serving_default'](x=tensor)
        e = time()
        # print(outs)
        raw_boxes = outs['output_0'][0].numpy()
        raw_scores = outs['output_1'][0].numpy()
        raw_class_ids = outs['output_2'][0].numpy()
        boxes, scores, class_ids = [], [], []

        indices = np.where(raw_scores > threshold)
        boxes = raw_boxes[indices]
        scores = raw_scores[indices]
        class_ids = raw_class_ids[indices].astype('uint8')
        boxes *= 640
        boxes = scale_boxes((640, 640), boxes, (h, w))

        class_names = [self.labels[int(c)] for c in class_ids]

        print([(list(b), s, c) for b, s, c in zip(boxes, scores, class_names)])

        # Arange box from top to bottom #
        indices = np.argsort(boxes[:, 3])
        corrected_boxes = [boxes[i] for i in indices] 
        corrected_classes = [class_names[i] for i in indices] 
        corrected_scores = [scores[i] for i in indices]
        #***#
        return e - s, corrected_boxes, corrected_scores, corrected_classes

def detect_single(img_path: str, confidence=0.5, view=0, save=0, save_path='output'):
    """
    Detect an image format both png and jpg
    :param img_path: str, path to image
    :param confidence: 0 <= confidence < 1, object's confidence
    :param view: int, 0-not show, 1-show image with drew box/dot, -1-show cut ROI
    :param save: int, 0-not save, 1-save image with drew box/dot, -1-save cut ROI
    :param save_path: str, folder to save image/roi
    :return: None
    """
    # ################ #
    # INFERENCE STEP   #
    # ################ #

    s = time()
    imp = img_path
    # imp = os.path.join(img_path, 'img_0.jpg')
    image = cv2.imread(imp)
    # Check if image is not read correctly, read by PIL 
    if image is None:
        im = Image.open(imp)
        image = np.array(im)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # print("Cannot read image %s"%img_path + " --> continue")
        # return
    h, w = image.shape[:2]
    src_im = image.copy()
    runtime, boxes, scores, labels = predictor(image, confidence)
    e1 = time()
    print("inference time: ", runtime)
    name = img_path.split('/')[-1]
    if not boxes:
        print("Cannot find any thing in %s !" %name )
        # if save:
        #     cv2.imwrite(os.path.join(save_path, name), pad_image)
        return
    print("Found in %s! Detect time: %.3f" % (name, (e1 - s)))

    # ##################### #
    # SHOW IMAGE            #
    # ##################### #
    if bool(view):
        draw = draw_boxes(src_im.copy(), boxes, scores, labels)
        key = show('prediction', draw)
        if key == ord('q'):
            exit(-1)
    if bool(save):
        to_xml(os.path.join(save_path, '.'.join(name.split('.')[:-1]) + '.xml'), name, boxes, labels, (w, h))

def detect_folder(folder_path: str, confidence=0.2, view=0, save=0, save_path='output'):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    names = os.listdir(folder_path)
    fullnames = [os.path.join(folder_path, name) for name in names if ('xml' not in name) and ('.DS' not in name)]
    for i, fn in enumerate(fullnames):
        print('**********', i, '**********')
        detect_single(fn, confidence, view, save, save_path)
    print("Done !")

if __name__ == '__main__':
    os.chdir("..")

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--path', type=str, 
                                # required=True, 
                                help='Path to the test data folder',
                                default=os.getcwd() + "/data/table_detection/yolo_yaml_data/test_infer")
    parser.add_argument('-t', '--threshold', type=float, default=0.5, help='Object detection threshold')
    parser.add_argument('-vf', '--view_image', type=int, default=0, help='1 for view image and 0 for not')
    parser.add_argument('-sf', '--save_image', type=int, default=1, help='1 for save image and 0 for not')
    parser.add_argument('-sp', '--save_path', type=str, default=os.getcwd() + '/VNG-table-detection/result',
                        help='specific save result folder when save_image=True')
    args = parser.parse_args()

    predictor = Detector()
    detect_folder(args.path, view=args.view_image, confidence=args.threshold,
                  save=args.save_image, save_path=args.save_path)