import tensorflow as tf
import numpy as np
import cv2
import json

class Model():
    ''' class for tflite model: (input: model_file_name) '''
    def __init__(self, model_path):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        print("initializing: %s ..." %(model_path))
        print("in/out tensor details:")
        print(self.input_details)
        print(self.output_details)

        self.interpreter.allocate_tensors()
        self.num_outputs = len(self.output_details)

    def getInputShape(self):
        ''' return model input shape '''
        input_shape = self.input_details[0]["shape"]
        return input_shape

    def runModel(self, img):
        ''' run tflite model with incoming img: (in: img, out: output tensors) '''
        self.interpreter.set_tensor(self.input_details[0]['index'], [img])
        self.interpreter.invoke()
        outputs = []
        for i in range(self.num_outputs):
            outputs.append(self.interpreter.get_tensor(self.output_details[i]['index']))
        return outputs

def hand_json(output_tensors, img_size, out_size):
    ''' output tensor, cam img size, model input size '''

    hand_joints = output_tensors[0][0].reshape(-1,2) #21,2
    hand_flag = output_tensors[1]

    #find ratio of input image and output tensor
    in_h = img_size[0]
    in_w = img_size[1]
    ratio_h = out_size[0] / in_h
    ratio_w = out_size[1] / in_w

    #append output x,y into list
    xs = []
    ys = []

    if hand_flag:
        for i in range(hand_joints.shape[0]):
            x = hand_joints[i,0] / ratio_w
            y = hand_joints[i,1] / ratio_h
            xs.append(x)    
            ys.append(y)

            #draw joint location on image
            # cv2.circle(img, (int(x), int(y)), THICKNESS * 2, POINT_COLOR, THICKNESS)
            # for connection in connections:
            #     x0, y0 = hand_joints[connection[0],:]
            #     x1, y1 = hand_joints[connection[1],:]
            #     x0 = x0 / ratio_w
            #     x1 = x1 / ratio_w
            #     y0 = y0 / ratio_h
            #     y1 = y1 / ratio_h
            #     #draw joint connections by line
            #     cv2.line(img, (int(x0), int(y0)), (int(x1), int(y1)), CONNECTION_COLOR, 1)

        #write output into json
        model_output_json = {}
        model_output_json['x'] = list(xs)
        model_output_json['y'] = list(ys)

        # with open("hand_joints.json","w") as json_file:
        #     json.dump(model_output_json, json_file)
        return model_output_json
    else:
        return None

def hair_json(output_tensors, threshold):
    mask = np.where(output_tensors[:,:,1] > threshold, 1, 0)
    mask_idx = np.nonzero(mask)

    #write output into json
    #append output x,y into list
    xs = []
    ys = []
    for i in range(len(mask_idx[0])):
        xs.append(str(mask_idx[0][i]))
        ys.append(str(mask_idx[1][i]))
    
    model_output_json = {}
    model_output_json['x'] = xs
    model_output_json['y'] = ys
    
    return model_output_json
