#!/usr/bin/python3.6

## import packages
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

import tensorflow as tf
tfconfig = tf.compat.v1.ConfigProto(allow_soft_placement=True)
tfconfig.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=tfconfig)

import sys
sys.path.append("./nn/conv")
from minigooglenet import MiniGoogLeNet
from resnet import ResNet
sys.path.append("./callbacks")
from trainingmonitor import TrainingMonitor

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report

from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from keras.optimizers import SGD
from keras.datasets import cifar10

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import argparse
import imutils


"""
# USAGE
- End-to-End training process, using MiniGoogLeNet to train cifar10
- Use Keras LearningRateScheduler & apply polynomial decay of learning rate
"""
# Params
EPOCHS = 1
INIT_LR = 1e-3

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
parser.add_argument("-o", "--output", required=True, \
        help="path to the output loss/accuracy plot")
args = vars(parser.parse_args())

assert os.path.exists(args["output"])


## Fetch dataset & preprocessing
print("[INFO] Fetch CIFAR-10 ....")
(trainX, trainY), (testX, testY) = cifar10.load_data()

trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0

# apply mean substraction
mean = np.mean(trainX, axis=0)
trainX -= mean
testX -= mean

# convert labels from ints to vectors
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

labelNames = ["airplane","automobile","bird","cat","deer","dog","frog",\
        "horse","ship","truck"]


# prepare model
# initalize data generator & apply data augmentation
aug = ImageDataGenerator(width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        fill_mode="nearest",
        )

# initialize call backs
figPath = os.path.sep.join([args["output"], "%d.png" % os.getpid()])
jsonPath = os.path.sep.join([args["output"], "%d.json" % os.getpid()])
callbacks = [TrainingMonitor(figPath, jsonPath=jsonPath), 
        LearningRateScheduler(poly_decay)]

# Initialize model & train & evaluate
print("[INFO] compiling model...")
sgd = SGD(lr=INIT_LR, momentum=0.9)
#model = MiniGoogLeNet.build(height=32, width=32, depth=3, classes=10)
model = ResNet.build(height=32, width=32, depth=3, classes=10, \
        stages=[3, 4, 6], filters=[64, 128, 256, 512], dataset="cifar")

model.compile(loss="categorical_crossentropy", optimizer=sgd, \
        metrics=["accuracy"])

print("[INFO] training model...")
H = model.fit_generator(
        aug.flow(trainX, trainY, batch_size=64), 
        validation_data=(testX, testY), 
        steps_per_epoch=len(trainX) // 64, 
        epochs=EPOCHS, 
        callbacks=callbacks,
        verbose=1,
        )

print("[INFO] seralizing model...")
modelPath = os.path.sep.join([args["output"], "%s_model.hdf5" % os.getpid()])
model.save(modelPath)

print("[INFO] evaluating model...")
predictions = model.predict(testX, batch_size=128)
print(classification_report(predictions.argmax(axis=1), testY.argmax(axis=1), \
        target_names=labelNames))



