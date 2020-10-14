import numpy as np
import os
import cv2
from tflite_model import *
import json

# Object detection imports
from object_detection.utils import label_map_util    ### CWH: Add object_detection path

# Model Preparation

# What model to download.
MODEL_NAME = 'ssd_mobilenet_v3_large_coco_2020_01_14'
PATH_TO_TFLITE = MODEL_NAME + '/model.tflite'
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('object_detection/data', 'mscoco_label_map.pbtxt') ### CWH: Add object_detection path

NUM_CLASSES = 90
model_tflite_ssd = Model(PATH_TO_TFLITE)
in_shape = model_tflite_ssd.getInputShape()
# Loading label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Helper code
def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

# added to put object in JSON
class Object(object):
    def toJSON(self):
        return json.dumps(self.__dict__)

def get_objects(image, threshold=0.5):
    
    imH, imW = image.shape[0], image.shape[1]
    image_pre = cv2.resize(image,(in_shape[2],in_shape[1]))
    outputs = model_tflite_ssd.runModel(image_pre)
    boxes, classes, scores = np.squeeze(outputs[0]),np.squeeze(outputs[1]).astype(np.uint8),np.squeeze(outputs[2])
    
    # ''' draw detection outputs on image'''
    # for i in range(len(scores)):
    #     if ((scores[i] > threshold) and (scores[i] <= 1.0)):

    #         # Get bounding box coordinates and draw box
    #         # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
    #         ymin = int(max(1,(boxes[i][0] * imH)))
    #         xmin = int(max(1,(boxes[i][1] * imW)))
    #         ymax = int(min(imH,(boxes[i][2] * imH)))
    #         xmax = int(min(imW,(boxes[i][3] * imW)))
    #         cv2.rectangle(image, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
    #         # Draw label
    #         object_name = category_index[int(classes[i]) + 1]['name'] # Look up object name from "labels" array using class index
    #         label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
    #         labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
    #         label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
    #         cv2.rectangle(image, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
    #         cv2.putText(image, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
    # cv2.imshow('Object detector', image)
    # cv2.waitKey(0)

    obj_above_thresh = sum(n > threshold for n in scores)
    print("detected %s objects in image above a %s score" % (obj_above_thresh, threshold))

    output = []

    # Add some metadata to the output
    item = Object()
    item.version = "0.0.1"
    item.numObjects = float(obj_above_thresh)
    item.threshold = threshold
    output.append(item)

    for c in range(0, len(scores)):
        class_name = category_index[classes[c] + 1]['name']
        if scores[c] >= threshold:      # only return confidences equal or greater than the threshold
            print(" object %s - score: %s, coordinates: %s" % (class_name, scores[c], boxes[c]))

            item = Object()
            item.name = 'Object'
            item.class_name = class_name
            item.score = float(scores[c])
            item.y = float(boxes[c][0])
            item.x = float(boxes[c][1])
            item.height = float(boxes[c][2])
            item.width = float(boxes[c][3])

            # item = Object()
            # item.name = 'Object'
            # item.class_name = class_name
            # item.score = format(float(scores[c]),'.2f')
            #
            # item.y = float(max(1,(boxes[c][0] * imH)))
            # item.x = float(max(1,(boxes[c][1] * imW)))
            # item.height = float(min(imH,(boxes[c][2] * imH)))
            # item.width = float(min(imW,(boxes[c][3] * imW)))

            output.append(item)

    outputJson = json.dumps([ob.__dict__ for ob in output])
    return outputJson
