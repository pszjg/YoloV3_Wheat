from PIL import Image, ImageDraw
import xml.etree.ElementTree as ET


def io_load_image(image_path):
    img = Image.open(image_path)
    img.show()
    return img


def io_display_image(image, width=0, height=0):
    if width > 0 and height > 0:
        image = image.resize((width, height))

    image.show()


def io_pil_size(image):
    channels = 1
    width, height = image.size
    if image.mode == "RGB":
        channels = 3
    return width, height, channels


def io_print_bounding(image, xml, thickness=6):
    data, labels = io_load_annotations(xml)
    for s in data[0]['object']:
        for i in range(thickness):
            ImageDraw.Draw(image).rectangle([int(s['xmin']) - int(i), int(s['ymin']) - int(i), int(s['xmax']) + int(i), int(s['ymax']) + int(i)], None, (255, 0, 0))
    return image


def io_load_annotations(annotatation):
    labels = []
    all_insts = []
    seen_labels = {}

    img = {'object': []}

    try:
        tree = ET.parse(annotatation)
    except Exception as e:
        print(e)
        print('Ignore this bad annotation: ' + annotatation)
        return

    for elem in tree.iter():
        if 'object' in elem.tag or 'part' in elem.tag:
            obj = {}

            for attr in list(elem):
                if 'name' in attr.tag:
                    obj['name'] = attr.text

                    if obj['name'] in seen_labels:
                        seen_labels[obj['name']] += 1
                    else:
                        seen_labels[obj['name']] = 1

                    if len(labels) > 0 and obj['name'] not in labels:
                        break
                    else:
                        img['object'] += [obj]

                if 'bndbox' in attr.tag:
                    for dim in list(attr):
                        if 'xmin' in dim.tag:
                            obj['xmin'] = int(round(float(dim.text)))
                        if 'ymin' in dim.tag:
                            obj['ymin'] = int(round(float(dim.text)))
                        if 'xmax' in dim.tag:
                            obj['xmax'] = int(round(float(dim.text)))
                        if 'ymax' in dim.tag:
                            obj['ymax'] = int(round(float(dim.text)))

    if len(img['object']) > 0:
        all_insts += [img]

    cache = {'all_insts': all_insts, 'seen_labels': seen_labels}

    return all_insts, seen_labels
