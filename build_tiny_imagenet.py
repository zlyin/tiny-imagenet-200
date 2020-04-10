#!/usr/bin/python3.6

## import packages
import os
import sys
sys.path.append("../Orca/io")
from hdf5datasetwriter import HDF5DatasetWriter
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imutils import paths
import numpy as np
import cv2
import json
from tqdm import tqdm

from config import tiny_imagenet_config as cfg



"""
Prepare tiny imagenet dataset;
Generate train/val/test hdf5 files;
"""
## grab trainPaths
trainPaths = list(paths.list_images(cfg.TRAIN_IMAGES))
trainLabels = [p.split(os.path.sep)[-3] for p in trainPaths]    # wordIDs
print(trainLabels[:10])

le = LabelEncoder()
trainLabels = le.fit_transform(trainLabels)

# stratify sampling test data out of trainPaths
trainPaths, testPaths, trainLabels, testLabels = train_test_split(trainPaths, \
        trainLabels, test_size=cfg.NUM_TEST_IMAGES, stratify=trainLabels,
        random_state=42)


## grab valPaths
# load mapping bwt val filenames and wordIDs
with open(cfg.VAL_MAPPINGS) as f:
    M = f.readlines()
M = [r.strip().split("\t")[:2] for r in M]
print(M[:1])

valPaths = [os.path.sep.join([cfg.VAL_IMAGES, m[0]]) for m in M]
valLabels = le.fit_transform([m[1] for m in M])


## construct a list pairing the training, validation, test image paths and,
## labels and hdf5 files
datasets = [
        ("train", trainPaths, trainLabels, cfg.TRAIN_HDF5),
        ("val", valPaths, valLabels, cfg.VAL_HDF5),
        ("test", testPaths, testLabels, cfg.TEST_HDF5),
        ]

# initialize RBG mean values
(R, G, B) = ([], [], [])

# loop over dataset & generate hdf5 file
for (dType, paths, labels, outputPath) in tqdm(datasets):
    # create HDF5 writer
    print("[INFO] building %s ..." % outputPath)
    writer = HDF5DatasetWriter(outputPath, (len(paths), 64, 64, 3))

    # progress bar
    for i, (path, label) in enumerate(tqdm(zip(paths, labels))):
        # load in image
        image = cv2.imread(path)

        # record RGB mean from training set
        if dType == "train":
            (b, g, r) = cv2.mean(image)[:3]
            R.append(r)
            G.append(g)
            B.append(b)

        # add image, label to HDF5 dataset
        writer.add([image], [label])
        pass
    
    # close HDF5 db when finish
    writer.close()

# serialize RBG mean values to json file
print("[INFO] serializing means ...")
DMean = {"R" : np.mean(R), "G" : np.mean(G), "B" : np.mean(B)}
with open(cfg.DATASET_MEAN, "w") as f:
    f.write(json.dumps(DMean))
f.close()

print("Done!")
