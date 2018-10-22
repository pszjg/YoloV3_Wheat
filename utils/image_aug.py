"""
    A work in progress
    Put together by Jonathon Gibbs (pszjg@nottingham.ac.uk)
    http://www.JonathonGibbs.com
    https://github.com/pszjg
    Thanks to numerous google sources for help
"""

import numpy as np
import random as rand
from PIL import Image, ImageOps, ImageEnhance, ImageFilter, ImageDraw
import sys
import inspect
from scipy.signal import convolve2d
import copy
import math


class ImageAugmentation:
    def __init__(self):
        self.categories = ['lighting']  # add transform color and distortion
        self.categories_limit = [2, 1, 1, 2, 2]
        self.operations = {}

        for i in range(len(self.categories)):
            self.operations[self.categories[i]] = []

        for name, obj in inspect.getmembers(sys.modules[__name__]):
            cat = name.split("_")[0]
            if cat in self.categories:
                self.operations[cat].append(name)

    def randomise_image(self, image):
        """
        Randomly apply augmentations to images
        :param image: A PIL image to be randomised
        :return: An augmented PIL Image
        """
        num_operations = rand.randint(0, 1)
        image_operations = []
        while num_operations > 0:
            searching = True
            while searching:
                category = rand.choice(self.categories)
                chosen_op = rand.choice(self.operations[category])
                if chosen_op not in image_operations:
                    if len(image_operations) > 0 and ("blur" in chosen_op or "edge" in chosen_op or "color" in chosen_op):
                        continue
                    image_operations.append(chosen_op)
                    method_call = getattr(sys.modules[__name__], chosen_op)
                    image = method_call(image)
                    searching = False
                num_operations = num_operations - 1
        return image


def convolve_all_colours(image, window):
    img = np.array(image)
    """
    Convolves image with window, over all three colour channels
    """
    ims = []
    for d in range(3):
        im_conv_d = convolve2d(img[:, :, d], window, mode="same", boundary="symm")
        ims.append(im_conv_d)

    im_conv = np.stack(ims, axis=2).astype("uint8")

    im = Image.fromarray(im_conv)
    return im


def color_reduce_channels(image):
    data = np.array(image)
    data[:, :, 0] = data[:, :, 0] - (data[:, :, 0] * 0.05)
    data[:, :, 1] = data[:, :, 1] - (data[:, :, 1] * 0.05)
    data[:, :, 2] = data[:, :, 2] - (data[:, :, 2] * 0.05)

    im = Image.fromarray(data)
    return im


def color_increase_channels(image):
    data = np.array(image)
    # data[:, :, 0] = data[:, :, 0] + (data[:, :, 0] * 0.05)
    data[:, :, 1] = data[:, :, 1] + (data[:, :, 1] * rand.uniform(0.05, 0.1))
    # data[:, :, 2] = data[:, :, 2] + (data[:, :, 2] * 0.05)

    im = Image.fromarray(data)
    return im


def color_swap_channels(image):
    channelorder = [1, 0, 2]
    rand.shuffle(channelorder)
    data = np.array(image)
    red, green, blue = data[:, :, 0], data[:, :, 1], data[:, :, 2]
    data[:, :, channelorder[0]] = red
    data[:, :, channelorder[1]] = green
    data[:, :, channelorder[2]] = blue

    im = Image.fromarray(data)
    return im


def color_sepia(image):
    data = np.array(image)
    red, green, blue = data[:, :, 0], data[:, :, 1], data[:, :, 2]

    data[:, :, 0] = 0.393 * red + 0.769 * green + 0.189 * blue
    data[:, :, 1] = 0.349 * red + 0.686 * green + 0.168 * blue
    data[:, :, 2] = 0.272 * red + 0.534 * green + 0.131 * blue

    np.clip(data[:, :, 0], 0, 255, data[:, :, 0])
    np.clip(data[:, :, 1], 0, 255, data[:, :, 1])
    np.clip(data[:, :, 2], 0, 255, data[:, :, 2])

    im = Image.fromarray(data)
    return im


def color_increase_channel(image, channel=rand.randint(0, 3), value=rand.uniform(-0.05, 0.05)):
    data = np.array(image)
    if channel == 3:
        for i in range(0, 2):
            data[:, :, i] = (data[:, :, i] * value) + data[:, :, i]
            np.clip(data[:, :, i], 0, 255, data[:, :, i])
    else:
        data[:, :, channel] = (data[:, :, channel] * value) + data[:, :, channel]
        np.clip(data[:, :, channel], 0, 255, data[:, :, channel])
    im = Image.fromarray(data)
    return im


def distortion_line(image, dimension=rand.randint(0, 2), occurance=rand.randint(10, 30), thickness=1):
    data = np.array(image)
    if dimension == 0 or dimension == 2:
        for i in range(0, np.size(data, 0), occurance):
            for j in range(thickness):
                    data[i - j, :, :] = [0, 0, 0]
    if dimension == 1 or dimension == 2:
        for i in range(0, np.size(data, 1), occurance):
            for j in range(thickness):
                    data[:, i - j, :] = [0, 0, 0]
    im = Image.fromarray(data)
    return im


def distortion_random_erase(image, size=rand.randint(1, 3), frequency=rand.randint(100, 200), color=(255, 255, 255)):
    for i in range(frequency):
        x = rand.randint(1, image.size[0] - 1)
        y = rand.randint(1, image.size[1] - 1)

        ImageDraw.Draw(image).rectangle([x, y, x + size, y + size], color)
    return image


def distortion_pixelate(image, size=rand.randint(100, 200)):
    # Resize smoothly down to 16x16 pixels
    img = image.resize((size, size), resample=Image.BILINEAR)
    # Scale back up using NEAREST to original size
    return img.resize(image.size, Image.NEAREST)


def distortion_sp(image, frequency=rand.randint(1000, 5000)):
    for i in range(frequency):
        x = rand.randint(1, image.size[0] - 1)
        y = rand.randint(1, image.size[1] - 1)

        s_p = rand.randint(0, 1)
        if s_p == 1:
            image.putpixel((x, y), (255, 255, 255))
        else:
            image.putpixel((x, y), (0, 0, 0))
    return image


def print_bounding_boxes(image, boxes, thickness=2):
    for i in range(len(boxes)):
        for j in range(thickness):
            ImageDraw.Draw(image).rectangle([int(boxes[i]['xmin']) - int(j), int(boxes[i]['ymin']) - int(j), int(boxes[i]['xmax']) + int(j), int(boxes[i]['ymax']) + int(j)], None, (255, 0, 0))

    return image


def lighting_saturation(image, level=rand.uniform(0.5, 1.5)):
    return ImageEnhance.Color(image).enhance(level)


def lighting_sharpness(image, level=rand.uniform(0.5, 1.5)):
    return ImageEnhance.Sharpness(image).enhance(level)


def lighting_brightness(image, level=rand.uniform(0.5, 1.5)):
    return ImageEnhance.Brightness(image).enhance(level)


def lighting_contrast(image, level=rand.uniform(0.5, 1.5)):
    return ImageEnhance.Contrast(image).enhance(level)


def _leave_lighting_gamma(image, gamma=rand.uniform(0.7, 1.3)):
    invert_gamma = 1.0 / gamma
    lut = [pow(x / 255., invert_gamma) * 255 for x in range(256)]
    lut = lut * 3
    image = image.point(lut)
    return image


def filters_gauss_blur(image, radius=rand.uniform(0.5, 1.5)):
    return image.filter(ImageFilter.GaussianBlur(radius))


def filters_unshaprp(image, radius=rand.randint(1, 8), percent=rand.randint(100, 200)):
    return image.filter(ImageFilter.UnsharpMask(radius, percent, 1))


def filters_edge(image):
    return image.filter(ImageFilter.EDGE_ENHANCE_MORE())


def filters_detail(image):
    return image.filter(ImageFilter.DETAIL())


def filters_autocontrast(image, threshold=rand.randint(1, 5)):
    return ImageOps.autocontrast(image, threshold)


def color_invert(image):
    return ImageOps.invert(image)


def transform_resize_image(image, width):
    val = max(image.size[0], image.size[1])
    val = val / width
    x = int(image.size[0] / val)
    y = int(image.size[1] / val)

    if x > width:
        x = width

    if y > width:
        y = width

    image = image.resize((x, y), Image.ANTIALIAS)

    return x, y, image


def transform_scale_image(image, scale=1.0):
    return image.resize((int(image.size[0] * scale), int(image.size[1] * scale)))


def transform_padding(image, padding=rand.randint(0, 100)):
    return ImageOps.expand(image, padding)


def transform_zoom(image, percentage=rand.uniform(0.8, 1.2)):
    img = Image.new('RGB', image.size, (255, 255, 255))
    image = image.resize((int(image.size[0] * percentage), int(image.size[1] * percentage)))
    img.paste(image, ((int((img.size[0] - image.size[0]) / 2)), int((img.size[1] - image.size[1]) / 2)))
    return img


def transform_flip(image):
    return ImageOps.mirror(image)


'''''''''''''''''''''''''''''''''''''''''''''
      Bounding box correction methods
'''''''''''''''''''''''''''''''''''''''''''''


def correct_bounding_boxes(boxes, new_w, new_h, net_w, net_h, flip, scale, image_w, image_h, angle):
    """
    A method to correctly orientate the boxes after image transformations have been applied

    :param boxes: An instance of boxes
    :param new_w: The new width of the image
    :param new_h: The new height of the image
    :param net_w: The width of the network (image)
    :param net_h: The height of the network (image)
    :param flip: Flip boolean (0,1)
    :param scale: The amount of scale (float)
    :param image_w: The original image width
    :param image_h: The original image height
    :param angle: The angle of rotation in degrees (0, 90, 180, 270)
    :return: Returns a series of correctly oriented boxes
    """

    boxes = copy.deepcopy(boxes)

    # randomize boxes' order
    np.random.shuffle(boxes)

    # correct sizes and positions
    sx, sy = float(new_w) / image_w, float(new_h) / image_h
    zero_boxes = []

    scale_x = 0
    scale_y = 0
    if scale != 1:
        scale_x = int((net_w - net_w * scale) / 2)
        scale_y = int((net_h - net_h * scale) / 2)

    for i in range(len(boxes)):
        boxes[i]['xmin'] = int(_constrain(0, net_w, (boxes[i]['xmin'] * sx) * scale + scale_x))
        boxes[i]['xmax'] = int(_constrain(0, net_w, (boxes[i]['xmax'] * sx) * scale + scale_x))
        boxes[i]['ymin'] = int(_constrain(0, net_h, (boxes[i]['ymin'] * sy) * scale + scale_y))
        boxes[i]['ymax'] = int(_constrain(0, net_h, (boxes[i]['ymax'] * sy) * scale + scale_y))

        if angle > 0:
            min_rot = rotate((int(net_w / 2), int(net_h / 2)), (boxes[i]['xmin'], boxes[i]['ymin']), (angle * 0.0174533) * -1)
            max_rot = rotate((int(net_w / 2), int(net_h / 2)), (boxes[i]['xmax'], boxes[i]['ymax']), (angle * 0.0174533) * -1)

            boxes[i]['xmin'] = min_rot[0]
            boxes[i]['ymin'] = min_rot[1]
            boxes[i]['xmax'] = max_rot[0]
            boxes[i]['ymax'] = max_rot[1]

            if boxes[i]['xmin'] > boxes[i]['xmax']:
                minx = boxes[i]['xmax']
                boxes[i]['xmax'] = boxes[i]['xmin']
                boxes[i]['xmin'] = minx
            if boxes[i]['ymin'] > boxes[i]['ymax']:
                miny = boxes[i]['ymax']
                boxes[i]['ymax'] = boxes[i]['ymin']
                boxes[i]['ymin'] = miny

        if scale != 1:
            if boxes[i]['xmax'] >= net_w or boxes[i]['xmax'] <= 0 or boxes[i]['ymax'] >= net_h or boxes[i]['ymax'] <= 0\
                    or boxes[i]['xmin'] >= net_w or boxes[i]['xmin'] <= 0 or boxes[i]['ymin'] >= net_h or boxes[i]['ymin'] <= 0\
                    or boxes[i]['xmax'] <= boxes[i]['xmin'] or boxes[i]['ymax'] <= boxes[i]['ymin']:
                zero_boxes += [i]
                continue

        if flip == 1:
            swap = boxes[i]['xmin']
            boxes[i]['xmin'] = net_w - boxes[i]['xmax']
            boxes[i]['xmax'] = net_w - swap

    boxes = [boxes[i] for i in range(len(boxes)) if i not in zero_boxes]

    return boxes


def _constrain(min_v, max_v, value):
    if value < min_v:
        return min_v
    if value > max_v:
        return max_v
    return value


def transform_rotate(image, angle):
    """
        The angle should be given in degrees.
    """
    image = image.rotate(angle, expand=0, center=(int(image.size[0]/2), int(image.size[1]/2))).resize(image.size)
    return image


def rotate(origin, point, angle):
    """
        The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy
