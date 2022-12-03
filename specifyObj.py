#!/usr/bin/env python
# coding: utf-8
"""
Object Detection (On Image) From TF2 Saved Model
=====================================
"""

import os
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

PATH_TO_LABELS = "exported-models/my_mobilenet_model/saved_model/label_map.pbtxt"



def specify(image, detectFn, confidence) :
      category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                        use_display_name=True)
      # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
      input_tensor = tf.convert_to_tensor(image)
      # The model expects a batch of images, so add an axis with `tf.newaxis`.
      input_tensor = input_tensor[tf.newaxis, ...]

      # input_tensor = np.expand_dims(image_np, 0)
      detections = detectFn(input_tensor)

      # All outputs are batches tensors.
      # Convert to numpy arrays, and take index [0] to remove the batch dimension.
      # We're only interested in the first num_detections.
      num_detections = int(detections.pop('num_detections'))

      detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
      detections['num_detections'] = num_detections

      # detection_classes should be ints.
      detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

      image_with_detections = image.copy()


      if (detections['detection_classes'][0] == 1) :
            print(detections['detection_scores'][0])
            print(detections['detection_boxes'][0])
            # detections['detection_boxes'][0] = [0.0612306,0.2669388,0.4139388,0.75616074]
            detections['detection_boxes'][0] = [0.2,0.4,0.6,0.9]



      # SET MIN_SCORE_THRESH BASED ON YOU MINIMUM THRESHOLD FOR DETECTIONS
      viz_utils.visualize_boxes_and_labels_on_image_array(
            image_with_detections,
            detections['detection_boxes'],
            detections['detection_classes'],
            detections['detection_scores'],
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=1,
            min_score_thresh=confidence,
            agnostic_mode=False)

      # return image_with_detections
      return (detections['detection_scores'][0], detections['detection_boxes'][0])


def loadModel(path):
      print('Loading model...', end='')
      start_time = time.time()

      specModel = tf.saved_model.load(path)

      end_time = time.time()
      elapsed_time = end_time - start_time
      print('Done! Took {} seconds'.format(elapsed_time))

      return specModel
