"""
Example usage:
    python create_robosub_tf_record.py
        --label_map_path=create_robosub_tf_record.pbtxt
        --data_dir=`pwd`
        --output_dir=`pwd`
"""

import hashlib
import io
import logging
import os
import random
import re

from lxml import etree
import xml.etree.ElementTree as ET
#import numpy as np
import PIL.Image
import tensorflow as tf

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util


flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'Root directory to raw pet dataset.')
flags.DEFINE_string('output_dir', '', 'Path to directory to output TFRecords.')
flags.DEFINE_string('label_map_path', 'auv_label_map.pbtxt',
                    'Path to label map proto')
FLAGS = flags.FLAGS

def dict_to_tf_example(data,
                       label_map_dict,
                       image_subdirectory,
                       ignore_difficult_instances=False):
  """Convert XML derived dict to tf.Example proto.

  Notice that this function normalizes the bounding box coordinates provided
  by the raw data.

  Args:
    data: dict holding PASCAL XML fields for a single image (obtained by
      running dataset_util.recursive_parse_xml_to_dict)
    not using: mask_path: String path to PNG encoded mask.
    label_map_dict: A map from string label names to integers ids.
    image_subdirectory: String specifying subdirectory within the
      Pascal dataset directory holding the actual image data.
    ignore_difficult_instances: Whether to skip difficult instances in the
      dataset  (default: False).
    not using: faces_only: If True, generates bounding boxes for pet faces. Otherwise
      generates bounding boxes (as well as segmentations for full pet bodies).

  Returns:
    example: The converted tf.Example.

  Raises:
    ValueError: if the image pointed to by data['filename'] is not a valid JPEG
  """
  img_path = os.path.join(image_subdirectory, data['filename'])
  with tf.gfile.GFile(img_path, 'rb') as fid:
    encoded_jpg = fid.read()
  encoded_jpg_io = io.BytesIO(encoded_jpg)
  image = PIL.Image.open(encoded_jpg_io)
  if image.format != 'JPEG':
    raise ValueError('Image format not JPEG')
  key = hashlib.sha256(encoded_jpg).hexdigest()

  width = int(data['size']['width'])
  height = int(data['size']['height'])

  xmins = []
  ymins = []
  xmaxs = []
  ymaxs = []
  classes = []
  classes_text = []
  truncated = []
  poses = []
  difficult_obj = []

  try:
      for obj in data['object']:
        difficult = bool(int(obj['difficult']))
        if ignore_difficult_instances and difficult:
          continue
        difficult_obj.append(int(difficult))

        xmins.append(float(obj['bndbox']['xmin']) / width)
        ymins.append(float(obj['bndbox']['ymin']) / height)
        xmaxs.append(float(obj['bndbox']['xmax']) / width)
        ymaxs.append(float(obj['bndbox']['ymax']) / height)

        class_name = obj['name']
        classes_text.append(class_name.encode('utf8'))
        classes.append(label_map_dict[class_name])
        truncated.append(int(obj['truncated']))
        poses.append(obj['pose'].encode('utf8'))
  except KeyError:
      pass

  example = tf.train.Example(features=tf.train.Features(feature={
    'image/height': dataset_util.int64_feature(height),
    'image/width': dataset_util.int64_feature(width),
    'image/filename': dataset_util.bytes_feature(
        data['filename'].encode('utf8')),
    'image/source_id': dataset_util.bytes_feature(
        data['filename'].encode('utf8')),
    'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
    'image/encoded': dataset_util.bytes_feature(encoded_jpg),
    'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
    'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
    'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
    'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
    'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
    'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
    'image/object/class/label': dataset_util.int64_list_feature(classes),
    'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
    'image/object/truncated': dataset_util.int64_list_feature(truncated),
    'image/object/view': dataset_util.bytes_list_feature(poses),
  }))
  return example

def create_tf_record(output_filename,
                     label_map_dict,
                     annotations_dir,
                     image_dir,
                     examples):
  writer = tf.python_io.TFRecordWriter(output_filename)

  for idx, example in enumerate(examples):
    if idx % 100 == 0:
        logging.info('On image %d of %d', idx, len(examples))
    xml_path = os.path.join(annotations_dir, 'xmls', example + '.xml')
    # mask_path = os.path.join(annotations_dir, 'trimaps', example + '.png')

    if not os.path.exists(xml_path):
      logging.warning('Could not find %s, ignoring example.', xml_path)
      continue
    with tf.gfile.GFile(xml_path, 'r') as fid:
      xml_str = fid.read()
    xml = etree.fromstring(xml_str)
    data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']

    tf_example = dict_to_tf_example(data, label_map_dict, image_dir)
    writer.write(tf_example.SerializeToString())

  writer.close()

def create_tf_example(example):
  # TODO(user): Populate the following variables from your example.
  height = None # Image height
  width = None # Image width
  filename = None # Filename of the image. Empty if image is not from file
  encoded_image_data = None # Encoded image bytes
  image_format = None # b'jpeg' or b'png'

  xmins = [] # List of normalized left x coordinates in bounding box (1 per box)
  xmaxs = [] # List of normalized right x coordinates in bounding box
             # (1 per box)
  ymins = [] # List of normalized top y coordinates in bounding box (1 per box)
  ymaxs = [] # List of normalized bottom y coordinates in bounding box
             # (1 per box)
  classes_text = [] # List of string class name of bounding box (1 per box)
  classes = [] # List of integer class id of bounding box (1 per box)

  tf_example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(filename),
      'image/source_id': dataset_util.bytes_feature(filename),
      'image/encoded': dataset_util.bytes_feature(encoded_image_data),
      'image/format': dataset_util.bytes_feature(image_format),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
  }))
  return tf_example


def main(_):
  # makes paths and reads examples list from trainval.txt
  data_dir = FLAGS.data_dir
  label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)
  logging.info('Reading from robosub dataset.')
  image_dir = os.path.join(data_dir, 'r_images')
  annotations_dir = os.path.join(data_dir, 'r_annotations')
  examples_path = os.path.join(annotations_dir, 'trainval.txt')
  examples_list = dataset_util.read_examples_list(examples_path)

  # Takes the list of examples and divides it into training and eval
  random.seed(42)
  random.shuffle(examples_list)
  num_examples = len(examples_list)
  # print "number of examples: " + str(num_examples)
  num_train = int(0.7 * num_examples)
  train_examples = examples_list[:num_train]
  val_examples = examples_list[num_train:]
  logging.info('%d training and %d validation examples.',
               len(train_examples), len(val_examples))

  train_output_path = os.path.join(FLAGS.output_dir, 'robosub_train.record')
  val_output_path = os.path.join(FLAGS.output_dir, 'robosub_val.record')


  create_tf_record(train_output_path, label_map_dict, annotations_dir,
                   image_dir, train_examples)
  create_tf_record(val_output_path, label_map_dict, annotations_dir,
                   image_dir, val_examples)


if __name__ == '__main__':
  tf.app.run()
