"""
config parameters for training
"""

import os
import sys


# define paths for training & validation directories
TRAIN_IMAGES = "./data/train"
VAL_IMAGES = "./data/val/images"

# define path to mapping btw val filenames and corresponding wordIDs
VAL_MAPPINGS = "./data/val/val_annotations.txt"

# define path to WordIDs & corresponding class labels
WORDNET_IDS = "./data/wnids.txt"
WORD_LABELS = "./data/words.txt"

# since we don't have test data, need to split a part of training data as test
NUM_CLASSES = 200
NUM_TEST_IMAGES = 50 * NUM_CLASSES

# define the path to the output training, validation & test HDF5 files
TRAIN_HDF5 = "./data/hdf5/train.hdf5"
VAL_HDF5 = "./data/hdf5/val.hdf5"
TEST_HDF5 = "./data/hdf5/test.hdf5"

# define the path to the dataset RGB mean
DATASET_MEAN = "./output/tiny_imagenet_200_mean.json"

# define the path to model outputs
OUTPUT_PATH = "./output"
#MODEL_PATH = os.path.sep.join([OUTPUT_PATH, "checkpoints/epoch_70.hdf5"])



