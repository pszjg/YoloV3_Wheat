from utils import dataloader, utils, generator
from model import createModel, callbacks
import os
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from tensorflow.python.client import device_lib

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='train and evaluate YOLO_v3 model on any dataset')
    argparser.add_argument('-c', '--conf', help='path to configuration file')

    args = argparser.parse_args()
    config_path = args.conf

    with open(config_path) as config_buffer:
        config = json.loads(config_buffer.read())

    # Load Annotated data
    train_ints, valid_ints, labels, max_box_per_image = dataloader.create_training_instances(
        config['train']['dataset'],
        config['train']['cache_name'],
        config['valid']['valid_annot_folder'],
        config['valid']['valid_image_folder'],
        config['valid']['cache_name'],
        config['model']['labels']
    )
    print('\nTraining on: \t' + str(labels) + '\n')

    # Generate series of batches
    train_set = generator.BatchGenerator(
        instances=train_ints,
        anchors=config['model']['anchors'],
        labels=labels,
        downsample=32,  # ratio between network input's size and network output's size, 32 for YOLOv3
        max_box_per_image=max_box_per_image,
        batch_size=config['train']['batch_size'],
        min_net_size=config['model']['min_input_size'],
        max_net_size=config['model']['max_input_size'],
        shuffle=True,
        jitter=0.2,
        norm=utils.normalize
    )

    valid_set = generator.BatchGenerator(
        instances=valid_ints,
        anchors=config['model']['anchors'],
        labels=labels,
        downsample=32,  # ratio between network input's size and network output's size, 32 for YOLOv3
        max_box_per_image=max_box_per_image,
        batch_size=config['train']['batch_size'],
        min_net_size=config['model']['min_input_size'],
        max_net_size=config['model']['max_input_size'],
        shuffle=True,
        jitter=0.0,
        norm=utils.normalize
    )

    # Create a model
    if os.path.exists(config['train']['saved_weights_name']):
        config['train']['warmup_epochs'] = 0
    warmup_batches = config['train']['warmup_epochs'] * (config['train']['train_times'] * len(train_set))

    # Set cuda environment
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"

    # Create the model
    model, infer_model = createModel.create_model(
        model = config['model']['type'],
        nb_class=len(labels),
        anchors=config['model']['anchors'],
        max_box_per_image=max_box_per_image,
        max_grid=[config['model']['max_input_size'], config['model']['max_input_size']],
        batch_size=config['train']['batch_size'],
        warmup_batches=warmup_batches,
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

    ###############################
    #   Kick off the training
    ###############################
    callbacks = callbacks.create_callbacks(config['train']['saved_weights_name'], config['train']['tensorboard_dir'], infer_model)

    history = model.fit_generator(
        generator=train_set,
        steps_per_epoch=len(train_set) * config['train']['train_times'],
        epochs=config['train']['nb_epochs'] + config['train']['warmup_epochs'],
        verbose=2 if config['train']['debug'] else 1,
        callbacks=callbacks,
        workers=4,
        max_queue_size=8
    )

    # grab the history object dictionary
    H = history.history

    # plot the training loss and accuracy
    N = np.arange(0, len(H["loss"]))
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(N, H["loss"], label="train_loss")
    plt.plot(N, H["val_loss"], label="test_loss")
    plt.plot(N, H["acc"], label="train_acc")
    plt.plot(N, H["val_acc"], label="test_acc")
    plt.title("MiniGoogLeNet on CIFAR-10")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()

    # save the figure
    plt.savefig(args["output"])
    plt.close()

    ###############################
    #   Run the evaluation
    ###############################
    # compute mAP for all the classes
    average_precisions = utils.evaluate(infer_model, valid_set)

    # print the score
    for label, average_precision in average_precisions.items():
        print(labels[label] + ': {:.4f}'.format(average_precision))
    print('mAP: {:.4f}'.format(sum(average_precisions.values()) / len(average_precisions)))