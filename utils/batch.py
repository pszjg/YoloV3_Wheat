import cv2
import numpy as np
from keras.utils import Sequence
from utils.bbox import BoundBox, bbox_iou
import utils.image_aug as aug
from utils.image_aug import ImageAugmentation
import random
from PIL import Image
import copy
import os

class GenerateBatch(Sequence):
    # Downsample: # ratio between network input's size and network output's size, 32 for YOLOv3
    def __init__(self, instances, anchors, labels, downsample=32, max_box_per_image=30, batch_size=1, min_net_size=320, max_net_size=608, shuffle=True, jitter=True, norm=None):
        self.instances = instances
        self.batch_size = batch_size
        self.labels = labels
        self.downsample = downsample
        self.max_box_per_image = max_box_per_image
        self.min_net_size = (min_net_size // self.downsample) * self.downsample
        self.max_net_size = (max_net_size // self.downsample) * self.downsample
        self.shuffle = shuffle
        self.jitter = jitter
        self.norm = norm
        self.anchors = [BoundBox(0, 0, anchors[2 * i], anchors[2 * i + 1]) for i in range(len(anchors) // 2)]
        self.net_h = 576
        self.net_w = 576
        self.perform_augmentation = ImageAugmentation()
        if shuffle:
            np.random.shuffle(self.instances)

    def __len__(self):
        return int(np.ceil(float(len(self.instances)) / self.batch_size))

    def __getitem__(self, idx):
        # get image input size, change every 10 batches
        net_h, net_w = self._get_net_size(idx)
        base_grid_h, base_grid_w = net_h // self.downsample, net_w // self.downsample

        # determine the first and the last indices of the batch
        l_bound = idx * self.batch_size
        r_bound = (idx + 1) * self.batch_size

        if r_bound > len(self.instances):
            r_bound = len(self.instances)
            l_bound = r_bound - self.batch_size

        x_batch = np.zeros((r_bound - l_bound, net_h, net_w, 3))  # input images
        t_batch = np.zeros((r_bound - l_bound, 1, 1, 1, self.max_box_per_image, 4))  # list of groundtruth boxes

        # initialize the inputs and the outputs
        yolo_1 = np.zeros((r_bound - l_bound, 1 * base_grid_h, 1 * base_grid_w, len(self.anchors) // 3, 4 + 1 + len(self.labels)))  # desired network output 1
        yolo_2 = np.zeros((r_bound - l_bound, 2 * base_grid_h, 2 * base_grid_w, len(self.anchors) // 3, 4 + 1 + len(self.labels)))  # desired network output 2
        yolo_3 = np.zeros((r_bound - l_bound, 4 * base_grid_h, 4 * base_grid_w, len(self.anchors) // 3, 4 + 1 + len(self.labels)))  # desired network output 3
        yolos = [yolo_3, yolo_2, yolo_1]

        dummy_yolo_1 = np.zeros((r_bound - l_bound, 1))
        dummy_yolo_2 = np.zeros((r_bound - l_bound, 1))
        dummy_yolo_3 = np.zeros((r_bound - l_bound, 1))

        instance_count = 0
        true_box_index = 0
        # do the logic to fill in the inputs and the output
        for train_instance in self.instances[l_bound:r_bound]:
            # augment input image and fix object's position and size
            img, all_objs = self._aug_image(train_instance, net_h, net_w)
            # print("/r" + train_instance['filename'])
            for obj in all_objs:
                # find the best anchor box for this object
                max_anchor = None
                max_index = -1
                max_iou = -1

                shifted_box = BoundBox(0, 0, obj['xmax'] - obj['xmin'], obj['ymax'] - obj['ymin'])

                for i in range(len(self.anchors)):
                    anchor = self.anchors[i]
                    iou = bbox_iou(shifted_box, anchor)

                    if max_iou < iou:
                        max_anchor = anchor
                        max_index = i
                        max_iou = iou

                # determine the yolo to be responsible for this bounding box
                yolo = yolos[max_index // 3]
                grid_h, grid_w = yolo.shape[1:3]

                # determine the position of the bounding box on the grid
                center_x = .5 * (obj['xmin'] + obj['xmax'])
                center_x = center_x / float(net_w) * grid_w  # sigma(t_x) + c_x
                center_y = .5 * (obj['ymin'] + obj['ymax'])
                center_y = center_y / float(net_h) * grid_h  # sigma(t_y) + c_y

                # determine the sizes of the bounding box
                w = np.log((obj['xmax'] - obj['xmin']) / float(max_anchor.xmax))  # t_w
                h = np.log((obj['ymax'] - obj['ymin']) / float(max_anchor.ymax))  # t_h

                box = [center_x, center_y, w, h]

                # determine the index of the label
                obj_indx = self.labels.index(obj['name'])

                # determine the location of the cell responsible for this object
                grid_x = int(np.floor(center_x))
                grid_y = int(np.floor(center_y))

                # assign ground truth x, y, w, h, confidence and class probs to y_batch
                yolo[instance_count, grid_y, grid_x, max_index % 3] = 0
                yolo[instance_count, grid_y, grid_x, max_index % 3, 0:4] = box
                yolo[instance_count, grid_y, grid_x, max_index % 3, 4] = 1.
                yolo[instance_count, grid_y, grid_x, max_index % 3, 5 + obj_indx] = 1

                # assign the true box to t_batch
                true_box = [center_x, center_y, obj['xmax'] - obj['xmin'], obj['ymax'] - obj['ymin']]
                t_batch[instance_count, 0, 0, 0, true_box_index] = true_box

                true_box_index += 1
                true_box_index = true_box_index % self.max_box_per_image

                # assign input image to x_batch
            if self.norm is not None:
                x_batch[instance_count] = self.norm(img)
            else:
                # plot image and bounding boxes for sanity check
                for obj in all_objs:
                    cv2.rectangle(img, (obj['xmin'], obj['ymin']), (obj['xmax'], obj['ymax']), (255, 0, 0), 2)
                    cv2.putText(img, obj['name'],
                                (obj['xmin'] + 2, obj['ymin'] + 12),
                                0, 1.2e-3 * img.shape[0],
                                (0, 255, 0), 1)

                x_batch[instance_count] = img

            # increase instance counter in the current batch
            instance_count += 1

        return [x_batch, t_batch, yolo_1, yolo_2, yolo_3], [dummy_yolo_1, dummy_yolo_2, dummy_yolo_3]

    def _get_net_size(self, idx):
        if idx % 10 == 0:
            net_size = self.downsample * np.random.randint(int(self.min_net_size / self.downsample), int(self.max_net_size / self.downsample + 1))
            # print("resizing: ", net_size, net_size)
            self.net_h, self.net_w = net_size, net_size
        return self.net_h, self.net_w

    def _aug_image(self, instance, net_h, net_w):
        image_name = instance['filename']
        boxes = copy.deepcopy(instance['object'])
        # Open image
        pil_image = Image.open(image_name).convert('RGB')

        # Perform random crop with upper bounds of 1152 - Temporarily commented out as performed cropping outside of application
        '''if pil_image.size[0] > 1152 and pil_image.size[1] > 1152:
            rand_x_width = 576
            rand_y_height = 576
            minx = random.randint(1, pil_image.size[0] - rand_x_width)
            miny = random.randint(1, pil_image.size[1] - rand_y_height)
        else:
            rand_x_width = 576
            rand_y_height = 576
            minx = random.randint(1, pil_image.size[0] - rand_x_width)
            miny = random.randint(1, pil_image.size[1] - rand_y_height)

        maxx = minx + rand_x_width
        maxy = miny + rand_y_height

        box = (minx, miny, maxx, maxy)
        pil_image = pil_image.crop(box)

        new_boxes = []

        for bounding in boxes:
            x_min = bounding['xmin']
            x_max = bounding['xmax']
            y_min = bounding['ymin']
            y_max = bounding['ymax']

            if x_min >= minx and x_min <= maxx and x_max >= minx and x_max <= maxx and y_min >= miny and y_min <= maxy and y_max >= miny and y_max <= maxy:
                temp_box = copy.deepcopy(bounding)
                temp_box['ymin'] = y_min - miny
                temp_box['xmin'] = x_min - minx
                temp_box['ymax'] = y_max - miny
                temp_box['xmax'] = x_max - minx
                new_boxes.append(temp_box)

        boxes = new_boxes'''

        # Original dimensions of image
        width_ = pil_image.size[0]
        height_ = pil_image.size[1]

        # Resize image to network
        new_w, new_h, pil_image = aug.transform_resize_image(pil_image, net_w)

        # Perform image augmentation
        if random.randint(0, 100) > 2:
            pil_image = self.perform_augmentation.randomise_image(pil_image)

        # Increase to network size
        image = Image.new('RGB', (net_w, net_h), (255, 255, 255))
        image.paste(pil_image, (0, 0))
        pil_image = image

        # Apply transformation augmentations
        scale = 1  # Scale performed by randomly cropping
        flip = random.randint(0, 1)
        rotate = random.randint(0, 25)

        # Scale
        pil_image = aug.transform_zoom(pil_image, scale)

        # Rotate
        rotation = 0
        if rotate == 1:
            rotation = random.randint(0, 3) * 90
            pil_image = aug.transform_rotate(pil_image, rotation)

        # Flip
        if flip == 1:
            pil_image = aug.transform_flip(pil_image)

        # Update bounding boxes
        all_objs = aug.correct_bounding_boxes(boxes, new_w, new_h, net_w, net_h, flip, scale, width_, height_, rotation)

        # Display bounding boxes on images
        # pil_image = aug.print_bounding_boxes(pil_image, all_objs)

        # Image to numpy array
        image = np.array(pil_image)
        image = image[:, :, ::-1]  # RGB image

        # name = os.path.basename(instance['filename'])
        # cv2.imwrite('./testing/printed/' + str(name) + '.jpg', image)

        return image, all_objs

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.instances)

    def num_classes(self):
        return len(self.labels)

    def size(self):
        return len(self.instances)

    def get_anchors(self):
        anchors = []

        for anchor in self.anchors:
            anchors += [anchor.xmax, anchor.ymax]

        return anchors

    def load_annotation(self, i):
        annots = []

        for obj in self.instances[i]['object']:
            annot = [obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax'], self.labels.index(obj['name'])]
            annots += [annot]

        if len(annots) == 0: annots = [[]]

        return np.array(annots)

    def load_image(self, i):
        return cv2.imread(self.instances[i]['filename'])
