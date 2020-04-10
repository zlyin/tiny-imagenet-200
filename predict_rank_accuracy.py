#!/usr/bin/python3.6

## import packages
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
import tensorflow as tf
tfconfig = tf.compat.v1.ConfigProto(allow_soft_placement=True)
tfconfig.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=tfconfig)

import sys
sys.path.append("../Orca/preprocessing")
from imagetoarraypreprocessor import ImageToArrayPreprocessor
from simplepreprocessor import SimplePreprocessor
from meanpreprocessor import MeanPreprocessor
sys.path.append("../Orca/io")
from hdf5datasetgenerator import HDF5DatasetGenerator
sys.path.append("../Orca/utils")
from rank5_accuracy import rank5_accuracy

from config import tiny_imagenet_config as cfg  
from keras.models import load_model
import json
import argparse


## arguments
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", required=True, help="path to a specific loaded model")
args = vars(parser.parse_args())
assert os.path.exists(args["model"])

## hyperparameters
BATCH = 64
MODEL_PATH = args["model"]


## load RGB means of training data
means = json.loads(open(cfg.DATASET_MEAN).read())

# image preprocessors
sp = SimplePreprocessor(64, 64)
mp = MeanPreprocessor(means["R"], means["G"], means["B"])
iap = ImageToArrayPreprocessor()

# create testGen
testGen = HDF5DatasetGenerator(cfg.TEST_HDF5, BATCH, \
        preprocessors=[sp, mp, iap], classes=cfg.NUM_CLASSES)

# load model & make predictions
print("[INFO] loading %s ..." % MODEL_PATH)
model = load_model(MODEL_PATH)

print("[INFO] predicting on the test data...")
predictions = model.predict_generator(
        testGen.generator(), 
        steps=testGen.numImages // BATCH,
        max_queue_size=BATCH * 2,
        )

# compute rank1 & rank5 accs
(rank1, rank5) = rank5_accuracy(predictions, testGen.db["labels"])
print("[INFO] rank-1 acc = {:.3f}%".format(rank1 * 100))
print("[INFO] rank-5 acc = {:.3f}%".format(rank5 * 100))

testGen.close()

print("[INFO] Done!")

