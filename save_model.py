import os

import tensorflow as tf
import yolo_core.utils as utils
from yolo_core.yolov4 import YOLO, decode, filter_boxes

from convert_config import config as default_config
from model_config import config as model_config
from yolo import model as yolo_model


def save_tf(config=default_config):
  STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(config)

  input_layer = tf.keras.layers.Input([config.input_size, config.input_size, 3])
  feature_maps = YOLO(input_layer, NUM_CLASS, config.model, config.tiny)
  bbox_tensors = []
  prob_tensors = []
  if config.tiny:
    for i, fm in enumerate(feature_maps):
      if i == 0:
        output_tensors = decode(fm, config.input_size // 16, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE, config.framework)
      else:
        output_tensors = decode(fm, config.input_size // 32, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE, config.framework)
      bbox_tensors.append(output_tensors[0])
      prob_tensors.append(output_tensors[1])
  else:
    for i, fm in enumerate(feature_maps):
      if i == 0:
        output_tensors = decode(fm, config.input_size // 8, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE, config.framework)
      elif i == 1:
        output_tensors = decode(fm, config.input_size // 16, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE, config.framework)
      else:
        output_tensors = decode(fm, config.input_size // 32, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE, config.framework)
      bbox_tensors.append(output_tensors[0])
      prob_tensors.append(output_tensors[1])
  pred_bbox = tf.concat(bbox_tensors, axis=1)
  pred_prob = tf.concat(prob_tensors, axis=1)
  if config.framework == 'tflite':
    pred = (pred_bbox, pred_prob)
  else:
    boxes, pred_conf = filter_boxes(pred_bbox, pred_prob, score_threshold=config.score_thres, input_shape=tf.constant([config.input_size, config.input_size]))
    pred = tf.concat([boxes, pred_conf], axis=-1)
  model = tf.keras.Model(input_layer, pred)
  utils.load_weights(model, config.weights, config.model, config.tiny)
  model.save(config.output)
  print("Saved model!")

  # reload model with new weights
  model_config.weights = config.output

  if config.classes:
    model_config.classes = config.classes

  yolo_model.set_config(model_config)
  print("done! reloading weights")
  #remove weights original file then
  os.remove(config.weights)
