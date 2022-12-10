#!/usr/bin/env python
# coding: utf-8
"""
Object Detection (On Image) From TF2 Saved Model
=====================================
"""

import os
import os.path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)

import pathlib
import tensorflow as tf
import cv2, time
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')   # Suppress Matplotlib warnings
from development.DetectedItem import *
 
PATH_TO_LABELS = "development/models/exported-models/my_mobilenet_model/saved_model/label_map.pbtxt"

class SubClassDetector :

      def __init__(self) -> None:
            itemsDetected = []

      def setPathToModel (self, path) :
            self.PathToModel = open(os.path.dirname(__file__) + '/' + path)

      def specify(self, image, startX, startY) :
            category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,use_display_name=True)                                   
            # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
            input_tensor = tf.convert_to_tensor(image)
            # The model expects a batch of images, so add an axis with `tf.newaxis`.
            input_tensor = input_tensor[tf.newaxis, ...]

            detections = self.detectFn(input_tensor)

            # All outputs are batches tensors.
            # Convert to numpy arrays, and take index [0] to remove the batch dimension.
            # We're only interested in the first num_detections.
            num_detections = int(detections.pop('num_detections'))

            detections = {key: value[0, :num_detections].numpy()
                        for key, value in detections.items()}
            detections['num_detections'] = num_detections

            # detection_classes store ints representing class ids
            detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

            #Element zero is the highest scoring in scan.
            # return image_with_detections
            self.item = DetectedItem(detections['detection_boxes'][0], 
             (category_index[int(detections['detection_classes'][0])])["name"],
              detections['detection_scores'][0], image, startX, startY)

      def loadModel(self, dir):
            print('Loading model...', end='')
            start_time = time.time()

            self.detectFn = tf.saved_model.load(dir)

            end_time = time.time()
            elapsed_time = end_time - start_time
            print('Done! Took {} seconds'.format(elapsed_time))
