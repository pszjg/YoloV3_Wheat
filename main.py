from utils import dataloader
import torch
import torch.utils.data as data_utils
from keras.applications.vgg16 import VGG16

if __name__ == '__main__':
    #     # Options
    dataset = 'Paragon2'
    boundingBox = 30
    numThreads = 4
    cuda = True
    train_batch = 6
    valid_batch = 1

    # Load data, or create dataset if it does not exist
    trainset = dataloader.datasetup(dataset, boundingBox, train=True)
    validset = dataloader.datasetup(dataset, boundingBox, train=False)

    # Output test samples of images
    #trainset.__test_data__(50, boundingBox)

    # Create multi process iterators over the dataset
    kwargs = {'num_workers': numThreads, 'pin_memory': True} if cuda else {}
    train_loader = data_utils.DataLoader(trainset, batch_size=train_batch, shuffle=True, **kwargs)
    valid_loader = data_utils.DataLoader(validset, batch_size=valid_batch, shuffle=True, **kwargs)

    # Output coordinates
    trainset.__write_xml__("./datasets/Paragon2", boundingBox)
    validset.__write_xml__("./datasets/Paragon2", boundingBox)

