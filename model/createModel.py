from model.YoloV3 import create_yolov3_model, dummy_loss
import os
from keras.optimizers import Adam, SGD, RMSprop
from colorama import init, Fore


def create_model(
    model,
    nb_class,
    anchors,
    max_box_per_image,
    max_grid, batch_size,
    warmup_batches,
    ignore_thresh,
    saved_weights_name,
    lr,
    grid_scales,
    obj_scale,
    noobj_scale,
    xywh_scale,
    class_scale,
    optimiser
):
    # Initialise color
    init()

    if model == "YoloV3":
        template_model, infer_model = create_yolov3_model(
            nb_class            = nb_class,
            anchors             = anchors,
            max_box_per_image   = max_box_per_image,
            max_grid            = max_grid,
            batch_size          = batch_size,
            warmup_batches      = warmup_batches,
            ignore_thresh       = ignore_thresh,
            grid_scales         = grid_scales,
            obj_scale           = obj_scale,
            noobj_scale         = noobj_scale,
            xywh_scale          = xywh_scale,
            class_scale         = class_scale
        )
    else:
        print(Fore.RED + "Error: model " + model + " not found. Currently support YoloV3" + Fore.RESET)
        return None

    # load the pretrained weight if exists, otherwise load the backend weight only
    if os.path.exists(saved_weights_name):
        print(Fore.GREEN + "\nLoading pretrained weights " + saved_weights_name + "\n" + Fore.RESET)
        template_model.load_weights(saved_weights_name)
    else:
        print(Fore.GREEN + "\nLoading default weights backend.h5\n" + Fore.RESET)
        template_model.load_weights("backend.h5", by_name=True)

    train_model = template_model

    if optimiser == "Adam":
        optimizer = Adam(lr=lr, clipnorm=0.001)
    elif optimiser == "SGD":
        optimizer = SGD(lr=5e-3, momentum=0.9)
    elif optimiser == "RMSProp":
        optimizer = RMSprop(0.001, 0.9, None)
    else:
        print(Fore.RED + "Error: Optimizer not found" + Fore.RESET)
        return None

    train_model.compile(loss=dummy_loss, optimizer=optimizer)

    return train_model, infer_model


