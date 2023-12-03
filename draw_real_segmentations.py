from __future__ import absolute_import, division

from configs import argHandler  # Import the default arguments
from utils import get_generator, alpha_blend, get_segmented_image
import os
import numpy as np
import cv2
from tqdm import tqdm
import pandas as pd
from PIL import Image

FLAGS = argHandler()
FLAGS.setDefaults()

WRITE_PATH = './data/real_OriginalImagesForMasks'
ANNOTATION_CSV_FILE = './data/radiology_segmentations.csv'

df = pd.read_csv(ANNOTATION_CSV_FILE)
try:
    os.makedirs(WRITE_PATH)
except:
    print("path already exists")

FLAGS.batch_size = 1
generator = get_generator(FLAGS.train_csv, FLAGS)

images_names = generator.get_images_names()

for batch_i in tqdm(range(generator.steps)):
    batch, y = generator.__getitem__(batch_i)
    if y[0] == 0:
        continue
    predicted_class = y[0]
    label = FLAGS.classes[predicted_class]

    image_path = os.path.join(FLAGS.image_directory, images_names[batch_i])
    original = cv2.imread(image_path.replace('_224', ''))  # load original image instead of 224 version

    cv2.imwrite(os.path.join(WRITE_PATH, images_names[batch_i]), original)
