from utils import data_loader
from model import model_create
import os, sys
import argparse
import json
import numpy as np
from utils.utils import get_yolo_boxes, makedirs
from utils.bbox import draw_boxes
import cv2
from colorama import init, Fore
import xml.etree.ElementTree as ET
from xml.dom import minidom
import shutil
from keras.utils import plot_model
from contextlib import redirect_stdout

def _main_(args):
    config_path = args.conf
    input_path = args.input
    output_path = args.output

    with open(config_path) as config_buffer:
        config = json.loads(config_buffer.read())

    makedirs(output_path)
    makedirs(output_path + "/annots/")
    makedirs(output_path + "/images/")
    makedirs(output_path + "/dl_annots/")

    ###############################
    #   Set some parameterW
    ###############################
    net_h, net_w = 576, 576  # a multiple of 32, the smaller the faster
    obj_thresh, nms_thresh = 0.2, 0.2

    # Load Annotated data
    train_ints, valid_ints, labels, max_box_per_image = data_loader.create_training_instances(
        config['train']['dataset'],
        config['train']['cache_name'],
        config['valid']['valid_annot_folder'],
        config['valid']['valid_image_folder'],
        config['valid']['cache_name'],
        config['model']['labels']
    )
    print(Fore.LIGHTGREEN_EX + 'Training on: \t' + Fore.RESET + str(labels) + '\n')

    # Set cuda environment
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"

    # Create the model
    model, infer_model = model_create.create_model(
        model=config['model']['type'],
        nb_class=len(labels),
        anchors=config['model']['anchors'],
        max_box_per_image=max_box_per_image,
        max_grid=[config['model']['max_input_size'], config['model']['max_input_size']],
        batch_size=config['train']['batch_size'],
        warmup_batches=0,
        ignore_thresh=config['train']['ignore_thresh'],
        saved_weights_name=config['train']['saved_weights_name'],
        lr=config['train']['learning_rate'],
        grid_scales=config['train']['grid_scales'],
        obj_scale=config['train']['obj_scale'],
        noobj_scale=config['train']['noobj_scale'],
        xywh_scale=config['train']['xywh_scale'],
        class_scale=config['train']['class_scale'],
        optimiser="Adam",
    )

    '''os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
    print(model.summary())
    with open('./modelsummary.txt', 'w') as f:
        with redirect_stdout(f):
            model.summary()
    plot_model(model, to_file='./model.png')
    input("Press Enter to continue...")'''

    ###############################
    #   Predict bounding boxes
    ###############################
    # do detection on an image or a set of images
    image_paths = []
    if os.path.isdir(input_path):
        for inp_file in os.listdir(input_path):
            image_paths += [input_path + inp_file]
    else:
        image_paths += [input_path]

    image_paths = [inp_file for inp_file in image_paths if (inp_file[-4:] in ['.jpg', '.png', 'JPEG', '.JPG'])]
    num_files = len(image_paths)
    counter = 1
    # the main loop
    for image_path in image_paths:
        file_name, file_ext = os.path.splitext(image_path)
        file_name = file_name.replace(input_path, "")
        
        sys.stdout.write("\r{0}".format(":: Processing image " + str(counter) + " / " + str(num_files)))
        sys.stdout.flush()
        counter += 1

        image = cv2.imread(image_path)
        image_h, image_w, _ = image.shape

        # predict the bounding boxes
        boxes = get_yolo_boxes(infer_model, [image], net_h, net_w, config['model']['anchors'], obj_thresh, nms_thresh)[0]

        # suppress non-maximal boxes
        do_nms(boxes, nms_thresh)

        # draw bounding boxes on the image using labels
        draw_boxes(image, boxes, config['model']['labels'], obj_thresh, image_w, True)

        # write the image with bounding boxes to file
        cv2.imwrite(output_path + "/dl_annots/" + file_name + ".jpg", np.uint8(image))

        _write_xml(output_path + "/images/", file_name + file_ext, output_path + "/annots/", file_name + ".xml", boxes, image_w, image_h, config['model']['labels'], obj_thresh)
        shutil.copy(image_path, output_path + "/images/" + file_name + file_ext)
    sys.stdout.write("\n")


def _write_xml(image_directory, filename, save_path, save_name, boxes, img_width, img_height, labels, obj_thresh):
    # Write xml
    # create the file structure
    annotation = ET.Element('annotation')
    items = ET.SubElement(annotation, 'folder')
    items.text = "images"

    item1 = ET.SubElement(annotation, 'filename')
    item1.text = filename

    item2 = ET.SubElement(annotation, 'path')
    item2.text = image_directory + filename

    item3 = ET.SubElement(annotation, 'source')
    item4 = ET.SubElement(item3, 'database')

    item5 = ET.SubElement(annotation, 'size')
    item6 = ET.SubElement(item5, 'width')
    item6.text = str(img_width)
    item7 = ET.SubElement(item5, 'height')
    item7.text = str(img_height)
    item8 = ET.SubElement(item5, 'depth')
    item8.text = str(3)

    item9 = ET.SubElement(annotation, 'segmented')
    item9.text = str(0)

    for box in boxes:
        label_str = ''
        label = -1

        for i in range(len(labels)):
            if box.classes[i] > obj_thresh:
                if label_str != '':
                    label_str += ', '
                label_str += (labels[i] + ' ' + str(round(box.get_score() * 100, 2)) + '%')
                label = i

        if label >= 0:
            item10 = ET.SubElement(annotation, 'object')
            item11 = ET.SubElement(item10, 'name')
            item12 = ET.SubElement(item10, 'pose')
            item13 = ET.SubElement(item10, 'truncated')
            item14 = ET.SubElement(item10, 'difficult')
            item15 = ET.SubElement(item10, 'bndbox')
            item16 = ET.SubElement(item15, 'xmin')
            item17 = ET.SubElement(item15, 'ymin')
            item18 = ET.SubElement(item15, 'xmax')
            item19 = ET.SubElement(item15, 'ymax')
            item11.text = 'Ear Tips'
            item12.text = "Unspecified"
            item13.text = str(0)
            item14.text = str(0)
            # Load bounding box data
            item16.text = str(box.xmin)
            item17.text = str(box.ymin)
            item18.text = str(box.xmax)
            item19.text = str(box.ymax)

    xmlstr = minidom.parseString(ET.tostring(annotation)).toprettyxml(indent="   ", encoding='UTF-8')
    with open(save_path + save_name, "w") as f:
        f.write(str(xmlstr.decode('UTF-8')))
        f.close()


def correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w):
    if (float(net_w) / image_w) < (float(net_h) / image_h):
        new_w = net_w
        new_h = (image_h * net_w) / image_w
    else:
        new_h = net_w
        new_w = (image_w * net_h) / image_h

    for i in range(len(boxes)):
        x_offset, x_scale = (net_w - new_w) / 2. / net_w, float(new_w) / net_w
        y_offset, y_scale = (net_h - new_h) / 2. / net_h, float(new_h) / net_h

        boxes[i].xmin = int((boxes[i].xmin - x_offset) / x_scale * image_w)
        boxes[i].xmax = int((boxes[i].xmax - x_offset) / x_scale * image_w)
        boxes[i].ymin = int((boxes[i].ymin - y_offset) / y_scale * image_h)
        boxes[i].ymax = int((boxes[i].ymax - y_offset) / y_scale * image_h)


def do_nms(boxes, nms_thresh):
    if len(boxes) > 0:
        nb_class = len(boxes[0].classes)
    else:
        return

    for c in range(nb_class):
        sorted_indices = np.argsort([-box.classes[c] for box in boxes])

        for i in range(len(sorted_indices)):
            index_i = sorted_indices[i]

            if boxes[index_i].classes[c] == 0:
                continue

            for j in range(i + 1, len(sorted_indices)):
                index_j = sorted_indices[j]

                if bbox_iou(boxes[index_i], boxes[index_j]) >= nms_thresh:
                    boxes[index_j].classes[c] = 0


def _interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b

    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2, x4) - x1
    else:
        if x2 < x3:
            return 0
        else:
            return min(x2, x4) - x3


def bbox_iou(box1, box2):
    intersect_w = _interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
    intersect_h = _interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])

    intersect = intersect_w * intersect_h

    w1, h1 = box1.xmax - box1.xmin, box1.ymax - box1.ymin
    w2, h2 = box2.xmax - box2.xmin, box2.ymax - box2.ymin

    union = w1 * h1 + w2 * h2 - intersect

    return float(intersect) / union


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='train and evaluate YOLO_v3 model on any dataset')
    argparser.add_argument('-c', '--conf', help='path to configuration file')
    argparser.add_argument('-i', '--input', help='path to an image, a directory of images, a video, or webcam')
    argparser.add_argument('-o', '--output', default='output/', help='path to output directory')

    # Initialise colorama
    init()

    _main_(argparser.parse_args())
