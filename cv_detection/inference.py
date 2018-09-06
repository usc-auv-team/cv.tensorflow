from __future__ import print_function

import numpy as np
import os
import sys
import tensorflow as tf
import timeit

import cv2

from collections import defaultdict
from io import StringIO
from PIL import Image

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("/home/pedro/usc-computer-vision-2018/tensorflow/models/research")
sys.path.append("/home/pedro/usc-computer-vision-2018/tensorflow/models/research/object_detection")

# ## Object detection imports
# Here are the imports from the object detection module.

from utils import label_map_util

from utils import visualization_utils as vis_util

# # Model preparation
# Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing `PATH_TO_CKPT` to point to a new .pb file.
# By default we use an "SSD with Mobilenet" model here. See the [detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.


# What model to download.
# MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
MODEL_NAME = '/home/pedro/usc-computer-vision-2018/tensorflow/models/research/exported_graphs/SSD_Mobilenet/new_data/4_16/step_200005'
# MODEL_NAME = '/home/pedro/pretrained_models/faster_rcnn_resnet101_coco_2018_01_28'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = '/home/pedro/usc-computer-vision-2018/tensorflow/models/research/robosub_label_map.pbtxt'

NUM_CLASSES = 13


def load_model():
    # ## Load a (frozen) Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    # ## Loading label map
    # Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine

    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                                use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    return detection_graph, category_index


# ## Helper code
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


def load_frame_into_numpy_array(frame):
    return np.array(frame).astype(np.uint8)


def main():
    detection_graph, category_index = load_model()

    start_time = timeit.default_timer()
    total_detection_time = 0
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            # vc = cv2.VideoCapture('/home/pedro/Videos/auv/GOPR0883_small.MP4')
            vc = cv2.VideoCapture(0)
            # fourcc = cv2.VideoWriter_fourcc(*'MP4V')
            # out = cv2.VideoWriter('/home/jaimin/Workspace/Wet Tests/output.MP4', fourcc, vc.get(cv2.CAP_PROP_FPS),
            #                       (int(vc.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))))

            vc.set(cv2.CAP_PROP_MODE, cv2.CAP_MODE_RGB)
            vc.set(cv2.CAP_PROP_FPS, 30)
            vc.set(cv2.CAP_PROP_CONVERT_RGB, 1)
            vc.set(cv2.CAP_PROP_ISO_SPEED, 400)

            print('Frame Width:', vc.get(cv2.CAP_PROP_FRAME_WIDTH))
            print('Frame Height:', vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print('Mode:', vc.get(cv2.CAP_PROP_MODE))
            print('FPS:', vc.get(cv2.CAP_PROP_FPS))
            print('Convert RGB:', vc.get(cv2.CAP_PROP_CONVERT_RGB))
            print('ISO Speed:', vc.get(cv2.CAP_PROP_ISO_SPEED))

            # Put the code in try-except statements
            # Catch the keyboard exception and
            # release the camera device and
            # continue with the rest of code.
            cv2.namedWindow('Camera Input', cv2.WINDOW_NORMAL)
            try:

                frame_count = 0
                while (vc.isOpened()):
                    # Capture frame-by-frame
                    ret, frame = vc.read()
                    if not ret:
                        # Release the Video Device if ret is false
                        vc.release()
                        # Message to be displayed after releasing the device
                        print("Released Video Resource")
                        break

                    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # the array based representation of the image will be used later in order to prepare the
                    # result image with boxes and labels on it.
                    image_np = load_frame_into_numpy_array(frame)

                    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                    image_np_expanded = np.expand_dims(image_np, axis=0)

                    frame_start = timeit.default_timer()

                    # Actual detection.
                    (boxes, scores, classes, num) = sess.run(
                        [detection_boxes, detection_scores, detection_classes, num_detections],
                        feed_dict={image_tensor: image_np_expanded})

                    total_detection_time += timeit.default_timer() - frame_start
                    frame_count += 1

                    # Visualization of the results of a detection.
                    vis_util.visualize_boxes_and_labels_on_image_array(
                        image_np,
                        np.squeeze(boxes),
                        np.squeeze(classes).astype(np.int32),
                        np.squeeze(scores),
                        category_index,
                        use_normalized_coordinates=True,
                        line_thickness=4)

                    # out.write(cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))

                    cv2.imshow('Camera Input', image_np)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                print('Total Time: ', timeit.default_timer() - start_time)
                print('Average Time :', total_detection_time / frame_count if frame_count > 0 else 0.0)

            except KeyboardInterrupt:
                # Release the Video Device
                vc.release()
                # Message to be displayed after releasing the device
                print("Released Video Resource")

    # out.release()
    cv2.destroyAllWindows()
    # print("Released Output Video")


if __name__ == '__main__':
    main()
