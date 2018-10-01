import torch.utils.data as data
from PIL import Image
import os
import os.path
import errno
import numpy as np
import torch
import codecs
import json
from .IO import render_heatmap_2d, get_image_loader
from torchvision.transforms.functional import to_pil_image, to_tensor
import gc
import math
import random
import PIL.ImageOps
import torchvision.transforms
import cv2
import shutil
import xml.etree.ElementTree as ET
import lxml.etree as etree
from xml.dom import minidom


class datasetup():

    def __init__(self, dataset, boundingbox, train=True):
        if dataset == 'Paragon2':
            self.folderName = dataset
            self.featureCount = 4
            self.featureThreshold = [0.2, 0.2, 0.2, 0.2]
            self.featureNames = ['Leaf Tips', 'Leaf Bases', 'Ear Tips', 'Ear Bases']
        else:
            print("Dataset not found")

        self.featureList = []
        self.boundingList = []
        self.featureCropCentre = []
        self.featureTransformed = []
        self.featureGTPoints = []
        self.boundingBoxSize = boundingbox
        self.imagefolder = './datasets/' + self.folderName + '/'
        self.datafile = os.path.join(os.path.expanduser('./datasets'), self.folderName + '.pt')
        self.train = train
        #self.args = args

        if not os.path.exists(self.datafile):
            print("Dataset file not found, initialising dataset for " + dataset)
            a = self._create_datafile(self.featureNames)

        a = torch.load(self.datafile)
        self.data = a['train'] if self.train else a['valid']

        imgFeatures = 0
        if self.train == True:
            print("Loading dataset " + dataset)
            for i in range(len(a['train'])):
                for j in range(len(self.featureNames)):
                    imgFeatures += len(a['train'][i]['annotation'][self.featureNames[j]])
            print("Total training annotated features: " + str(imgFeatures))
        else:
            for i in range(len(a['valid'])):
                for j in range(len(self.featureNames)):
                    imgFeatures += len(a['valid'][i]['annotation'][self.featureNames[j]])
            print("Total valid annotated features: " + str(imgFeatures))

        self.imagechannelcount = 3
        self.channelcount = self.featureCount
        self.channelthresholds = self.featureThreshold
        self.load_image = get_image_loader()

    def _create_datafile(self, featureNames):
        print("Generating dataset cache.")
        allfiles = []
        for root, dirs, files in os.walk(self.imagefolder):
            for filename in files:
                if filename.endswith(('.jpg', '.JPG', '.png', '.PNG', '.JPEG')):
                    allfiles.append(filename)

        imagecount = len(allfiles)
        traincount = int(round(imagecount * 0.8))
        validcount = imagecount - traincount

        shuffle = torch.randperm(imagecount)
        trainidx = shuffle[0:traincount]
        valididx = shuffle[traincount:]

        traindata = []

        # All training files
        print("Scanning training data...")
        for idx in trainidx:
            filename = allfiles[idx]
            dt, bb = self._scanfile(os.path.join(self.imagefolder, filename), featureNames)
            if dt:
                traindata.append({'filename': filename, 'annotation': dt, 'bounding': bb})

        print("Validation data size: ", len(traindata))

        validdata = []

        print("Scanning validation data...")
        for idx in valididx:
            filename = allfiles[idx]
            dt, bb = self._scanfile(os.path.join(self.imagefolder, filename), featureNames)
            if dt:
                validdata.append({'filename': filename, 'annotation': dt, 'bounding': bb})

        print("Validation data size: ", len(validdata))
        print("Saving data...")

        cachedata = {
            'train': traindata,
            'valid': validdata
        }

        torch.save(cachedata, self.datafile)
        return self.datafile

    def __getitem_output__(self, image_data, boundingbox=0):
        self.featureLength = len(self.featureNames)
        self.featureList = []
        self.boundingList = []

        print("Printing sample file: " + str(image_data['filename']))

        # Index is adjusted so that the image image is included multiple times
        image_drawing = cv2.imread(os.path.join(self.imagefolder, image_data['filename']), 1)

        # Get features
        for i in range(len(self.featureNames)):
            self.featureList.append(image_data['annotation'][self.featureNames[i]].clone())
            self.boundingList.append(image_data['bounding'][self.featureNames[i]].clone())
            for j in range(len(self.featureList[i])):
                image_drawing = cv2.rectangle(image_drawing, (int(self.boundingList[i][j][0]), int(self.boundingList[i][j][1])), (int(self.boundingList[i][j][2]), int(self.boundingList[i][j][3])), (0, 0, 255), 5)
                image_drawing = cv2.rectangle(image_drawing, (int(self.boundingList[i][j][0] - 50), int(self.boundingList[i][j][1] - 25)), (int(self.boundingList[i][j][2] + 50), int(self.boundingList[i][j][1])), (120, 120, 120), 25)
                cv2.putText(image_drawing, self.featureNames[i] + " (1.00)", (int(self.boundingList[i][j][0] - 45), int(self.boundingList[i][j][1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
        cv2.imwrite('C:/Users/Jono/Documents/DeepLearning/testing/' + image_data['filename'] + '.png', image_drawing)

    def __write_xml__(self, image_directory, boundingbox):
        print("Moving files for dataset organisation")
        for i in range(len(self.data)):
            filename = self.data[i]['filename']
            if filename.endswith(".JPG") or filename.endswith(".jpg") or filename.endswith(".png"):
                image_drawing = cv2.imread(os.path.join(self.imagefolder, filename), 1)
                # Move images
                shutil.move(image_directory + "/" + filename, image_directory + "/images/" + filename)
                # Move JSON files
                shutil.move(image_directory + "/" + os.path.splitext(filename)[0] + '.json', image_directory + "/json/" + os.path.splitext(filename)[0] + '.json')
                # Get bounding box and write xml files
                ft, bb = self._scanfile(image_directory + "/json/" + os.path.splitext(filename)[0] + '.json', self.featureNames)
                # Write xml
                # create the file structure
                annotation = ET.Element('annotation')
                items = ET.SubElement(annotation, 'folder')
                items.text = "images"

                item1 = ET.SubElement(annotation, 'filename')
                item1.text = filename

                item2 = ET.SubElement(annotation, 'path')
                item2.text = image_directory + "/images/" + filename

                item3 = ET.SubElement(annotation, 'source')
                item4 = ET.SubElement(item3, 'database')

                item5 = ET.SubElement(annotation, 'size')
                image = cv2.imread(image_directory + "/images/" + filename)
                item6 = ET.SubElement(item5, 'width')
                item6.text = str(np.size(image, 1))
                item7 = ET.SubElement(item5, 'height')
                item7.text = str(np.size(image, 0))
                item8 = ET.SubElement(item5, 'depth')
                item8.text = str(3)


                item9 = ET.SubElement(annotation, 'segmented')
                item9.text = str(0)

                self.featureList = []
                self.boundingList = []

                for j in range(len(self.featureNames)):
                    self.featureList.append(self.data[i]['annotation'][self.featureNames[j]].clone())
                    self.boundingList.append(self.data[i]['bounding'][self.featureNames[j]].clone())

                for y in range(len(self.featureNames)):
                    for j in range(len(self.boundingList[y])):
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
                        item11.text = self.featureNames[y]
                        item12.text = "Unspecified"
                        item13.text = str(0)
                        item14.text = str(0)
                        bb1 = int(self.boundingList[y][j][0] * random.uniform(0.99, 1.01))
                        bb2 = int(self.boundingList[y][j][1] * random.uniform(0.99, 1.01))
                        bb3 = int(self.boundingList[y][j][2] * random.uniform(0.99, 1.00))
                        bb4 = int(self.boundingList[y][j][3] * random.uniform(0.99, 1.00))
                        item16.text = str(bb1)
                        item17.text = str(bb2)
                        item18.text = str(bb3)
                        item19.text = str(bb4)
                        image_drawing = cv2.rectangle(image_drawing, (bb1, bb2), (bb3, bb4), (0, 0, 255), 5)
                        cv2.putText(image_drawing, self.featureNames[y] + " (1.00)", (bb1 - 45, bb2 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)

                xmlstr = minidom.parseString(ET.tostring(annotation)).toprettyxml(indent="   ", encoding='UTF-8')
                with open(image_directory + "/annots/" + os.path.splitext(filename)[0] + '.xml', "w") as f:
                    f.write(str(xmlstr.decode('UTF-8')))
                    f.close()

                cv2.imwrite('C:/Users/Jono/Documents/DeepLearning/testing/' + filename + '.png', image_drawing)


    def __test_data__(self, outputs, boundingbox):
        for i in range(0, outputs):
            index = random.randint(0, 500)
            self.__getitem_output__(self.data[index], boundingbox)

    def __len__(self):
        return len(self.data)

    def _scanfile(self, path, featureNames):
        jsonpath = os.path.splitext(path)[0] + '.json'

        if os.path.isfile(jsonpath):
            with open(jsonpath) as data_file:
                jsondata = json.load(data_file)

            metadata = jsondata['metadata']
            data = jsondata['data']

            if metadata['Ignore']:
                return None, None

            featurePoints = []
            boundingBoxes = []
            for i in range(len(featureNames)):
                featurePoints.append([])
                boundingBoxes.append([])

            for value in data:
                t = value['Type']
                n = value['Name']
                # Currently handles only one point out of a possible list
                if t == "Point":
                    row = value['Points'][0].split(',')
                    p = [float(row[0]), float(row[1])]

                    for i in range(len(featureNames)):
                        if n == featureNames[i]:
                            featurePoints[i].append(p)
                            if self.boundingBoxSize >= 0:
                                b = float(p[0] - self.boundingBoxSize), float(p[1] - self.boundingBoxSize), float(p[0] + self.boundingBoxSize), float(p[1] + self.boundingBoxSize)
                                boundingBoxes[i].append(b)
                            else:
                                boundingBoxes[i].append(float(0))

            featureCount = 0
            for i in range(len(featureNames)):
                featureCount = featureCount + len(featurePoints[i])

            if featureCount == 0:
                return None, None

            returnFeatures = {}
            returnBounding = {}
            for i in range(len(featureNames)):
                returnFeatures[featureNames[i]] = torch.FloatTensor(featurePoints[i])
                returnBounding[featureNames[i]] = torch.FloatTensor(boundingBoxes[i])

            return returnFeatures, returnBounding

        return None, None

