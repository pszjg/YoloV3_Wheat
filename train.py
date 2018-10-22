from utils import data_loader, utils, batch
from model import model_create, callbacks_create
import os
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from colorama import init, Fore


def _main_(args):
    config_path = args.conf

    with open(config_path) as config_buffer:
        config = json.loads(config_buffer.read())

    # Initialise colorama
    init()

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

    # Generate series of batches
    training_set = batch.GenerateBatch(
        instances=train_ints,  # Images of training set
        anchors=config['model']['anchors'],
        labels=labels,
        downsample=32,  # ratio between network input's size and network output's size, 32 for YOLOv3
        max_box_per_image=max_box_per_image,
        batch_size=config['train']['batch_size'],
        min_net_size=config['model']['min_input_size'],
        max_net_size=config['model']['max_input_size'],
        shuffle=True,
        jitter=0.05,
        norm=utils.normalize
    )

    validiation_set = batch.GenerateBatch(
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
    warmup_batches = config['train']['warmup_epochs'] * (config['train']['train_times'] * len(training_set))

    # Create the model
    model, infer_model = model_create.create_model(
        model=config['model']['type'],
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
        optimiser=config['train']['optimizer'],
    )

    ###############################
    #   Kick off the training
    ###############################
    callbacks = callbacks_create.create_callbacks(
        config['train']['saved_weights_name'],
        config['train']['tensorboard_dir'],
        infer_model
    )

    history = model.fit_generator(
        generator=training_set,
        steps_per_epoch=len(training_set) * config['train']['train_times'],
        epochs=config['train']['nb_epochs'] + config['train']['warmup_epochs'],
        verbose=config['train']['debug'],
        callbacks=callbacks,
        workers=4,
        max_queue_size=8
    )

    ###############################
    #   Run the evaluation
    ###############################
    # compute mAP for all the classes
    average_precisions = utils.evaluate(infer_model, validiation_set)

    # print the score
    for label, average_precision in average_precisions.items():
        print(labels[label] + ': {:.4f}'.format(average_precision))
    print('mAP: {:.4f}'.format(sum(average_precisions.values()) / len(average_precisions)))

    # grab the history object dictionary
    hist = history.history
    # plot the training loss and accuracy
    numpy_plot = np.arange(0, len(hist["loss"]))
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(numpy_plot, hist["loss"], label="train_loss")
    plt.title("Training Results")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()

    # save the figure
    plt.savefig("D:/0.Projects/YoloV3_Wheat/loss.png")
    plt.close()


if __name__ == '__main__':
    # Disable tensorFlow debug information
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # Set cuda environment
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"

    argparser = argparse.ArgumentParser(description='train and evaluate YOLO_v3 model on any dataset')
    argparser.add_argument('-c', '--conf', help='path to configuration file')
    argparser.add_argument('-o', '--output', help='path to output directory')

    _main_(argparser.parse_args())
