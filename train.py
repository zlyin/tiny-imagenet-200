#!/usr/bin/python3.6

## import packages
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
tfconfig = tf.compat.v1.ConfigProto(allow_soft_placement=True)
tfconfig.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=tfconfig)

import sys
sys.path.append("../Orca/nn/conv")
from resnet import ResNet, ResNet50
from deepergooglenet import DeeperGoogLeNet
sys.path.append("../Orca/callbacks")
from trainingmonitor import TrainingMonitor
from epochcheckpoint import EpochCheckpoint
from parallelcheckpoint import ParallelCheckpoint
sys.path.append("../Orca/preprocessing")
from simplepreprocessor import SimplePreprocessor
from meanpreprocessor import MeanPreprocessor
from imagetoarraypreprocessor import ImageToArrayPreprocessor
sys.path.append("../Orca/io")
from hdf5datasetgenerator import HDF5DatasetGenerator

from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from keras.optimizers import SGD
from keras.models import load_model
from keras.utils import multi_gpu_model

from config import tiny_imagenet_config as cfg
import argparse
import json


"""
# USAGE
- End-to-End training process, using ResNet to train tiny-imagenet
- Use Keras LearningRateScheduler & apply polynomial decay of learning rate
"""
# Params
EPOCHS = 20
INIT_LR = 1e-4
BATCH = 128 * 2

def poly_decay(epoch):
    """
    polynomial learning rate decay: alpha = alpha0 * (1 - epoch/num_epochs) ** p
    - alpha0 = initial learning rate
    - p = exp index, can be 1, 2, 3 ... etc
    - epoch = current epoch number of training process
    """
    maxEpochs = EPOCHS
    baseLR = INIT_LR
    power = 1.0

    # compute
    lr = baseLR * (1 - (epoch / float(maxEpochs))) ** power
    return lr


## Build arguments parser
parser = argparse.ArgumentParser()
parser.add_argument("-c", "--checkpoints", required=True, help="path to the model weights & learning curves")
parser.add_argument("-m", "--model", type=str, help="path to a specific loaded model")
parser.add_argument("-s", "--start_epoch", type=int, default=0, help="epoch to start training from")
args = vars(parser.parse_args())
assert os.path.exists(args["checkpoints"])


## prepare model
# initalize data generator & apply data augmentation
aug = ImageDataGenerator(
        rotation_range=18,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest",
        )

# create preprocessors
means = json.loads(open(cfg.DATASET_MEAN, "r").read())
mp = MeanPreprocessor(means["R"], means["G"], means["B"])
sp = SimplePreprocessor(64, 64)
iap = ImageToArrayPreprocessor()

# initiate data genrators
trainGen = HDF5DatasetGenerator(cfg.TRAIN_HDF5, 64, aug=aug, preprocessors=[sp,
    mp, iap], classes=cfg.NUM_CLASSES)
valGen = HDF5DatasetGenerator(cfg.VAL_HDF5, 64, aug=aug, preprocessors=[sp,
    mp, iap], classes=cfg.NUM_CLASSES)
testGen = HDF5DatasetGenerator(cfg.TEST_HDF5, 64, preprocessors=[sp, mp, iap], 
        classes=cfg.NUM_CLASSES)

## build model
#opt = Adam(lr=INIT_LR)
opt = SGD(lr=INIT_LR, momentum=0.9)

if args["model"] is None:
    print("[INFO] compiling parallel model...")
    # exp1
    with tf.device("/cpu:0"):
        single_model = ResNet.build(64, 64, 3, cfg.NUM_CLASSES, [3, 4, 6], [64, 128, 256, 512], 
                reg=5e-4, bnEps=2e-5, bnMom=0.9, dataset="tiny-imagenet")
    
    # create distribute strategy for TF2.0
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = multi_gpu_model(single_model, gpus=2)
        model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

else:
    print("[INFO] loading %s ..." % args["model"])
    with tf.device("/cpu:0"):
        single_model = load_model(args["model"])

    # create distribute strategy for TF2.0
    new_strategy = tf.distribute.MirroredStrategy()
    with new_strategy.scope():
        model = multi_gpu_model(single_model, gpus=2)
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=["accuracy"])

    # update learning rate to a smaller one
    print("[INFO] old learning rate =", K.get_value(model.optimizer.lr))
    K.set_value(model.optimizer.lr, INIT_LR)
    print("[INFO] new learning rate =", K.get_value(model.optimizer.lr))

# set up callbacks
FIG_PATH = os.path.sep.join([args["checkpoints"], "resnet_tiny-imagenet.png"])
JSON_PATH = os.path.sep.join([args["checkpoints"], "resnet_tiny-imagenet.json"])

callbacks = [
        ParallelCheckpoint(single_model, args["checkpoints"], every=5, startAt=args["start_epoch"]), 
        TrainingMonitor(FIG_PATH, jsonPath=JSON_PATH, startAt=args["start_epoch"]),
        #LearningRateScheduler(poly_decay),  # Exp4
        ]

# train & evaluate
print("[INFO] training model...")
H = model.fit_generator(
        trainGen.generator(),
        steps_per_epoch=trainGen.numImages // BATCH,
        validation_data=valGen.generator(),
        validation_steps=valGen.numImages // BATCH, 
        epochs=EPOCHS, 
        max_queue_size=BATCH * 2,
        callbacks=callbacks,
        verbose=1,
        )
# close
trainGen.close()
valGen.close()


print("[INFO] Done!")


