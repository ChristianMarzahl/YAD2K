"""
This is a script that can be used to retrain the YOLOv2 model for your own dataset.
"""
import argparse

import os

import matplotlib.pyplot as plt
import numpy as np
import pickle
from PIL import Image
import cv2
import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Lambda, Conv2D
from keras.models import load_model, Model
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import Iterator

from yad2k.models.keras_yolo import (preprocess_true_boxes, yolo_body,
                                     yolo_eval, yolo_head, yolo_loss)
from yad2k.utils.draw_boxes import draw_boxes

from xml.dom.minidom import parse
import xml.dom.minidom
from tqdm import tqdm

import glob
import random
import os


target_width = 416*2
target_height = 416*2

class ImageDataGenerator(Iterator):

    def __init__(self, data_path, batch_size,anchors, shuffle=True, seed=None):

        with open(data_path, mode='rb') as f:
            data = pickle.load(f)


        self.image_paths =   data['images']
        self.boxes = data['boxes']
        self.anchors = anchors


        return super().__init__(len(self.image_paths), batch_size, shuffle, seed)

    def next(self):

        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)

        batch_image_paths = np.array(self.image_paths)[index_array]
        batch_boxes = np.array(self.boxes)[index_array]

        image_data, boxes = process_data(batch_image_paths, batch_boxes)

        detectors_mask, matching_true_boxes = get_detector_mask(boxes, self.anchors)

        return [image_data, boxes, detectors_mask, matching_true_boxes], np.zeros(len(image_data))


class ImageDataGeneratorXML_RegionCNN(Iterator):
    """description of class"""

    def __init__(self, xml_paths, batch_size,anchors, operations_dict = None, shuffle=True, seed=None):

        self.anchors = anchors

        self.image_paths = []
        self.boxes = []
        for xml_path in tqdm(xml_paths):
            image_path, image_boxes = self._convert_xml(xml_path)

            self.image_paths.append(image_path)
            self.boxes.append(image_boxes)

        self.image_paths = np.array(self.image_paths)
        self.boxes = np.array(self.boxes)

        self.operations_dict = operations_dict
        if self.operations_dict is None:
            self.operations_dict = {}

        return super().__init__(len(self.image_paths), batch_size, shuffle, seed)
            
    def next(self):

        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)

        batch_image_paths = self.image_paths[index_array]
        batch_boxes = self.boxes[index_array]

        result_images = []
        result_boxes = []
        
        for image_path, image_boxes in zip(batch_image_paths,batch_boxes):
            image_boxes = image_boxes.copy()


            basic_path = "C:/Users/Bronzi/Downloads/Train/"
            image_path = os.path.join(basic_path,image_path.split("\\")[-1])
            image = cv2.imread(image_path) # replace with defalut image loader

            for operation_key, operation_value  in self.operations_dict.items():

                # check if the selected operation should be performed 
                if np.random.choice([1,0],1)[0] == 1:
                    continue

                if operation_key.lower() == "flip".lower():
                    image, image_boxes = flip_image(image,image_boxes,np.random.choice(operation_value,1)[0])
                    continue

                if operation_key.lower() == "random_shift".lower() and np.random.choice([1,0],1)[0] == 1:
                    image, image_boxes = random_shift(image,operation_value["max_x_shift"],operation_value["max_y_shift"],image_boxes)
                    continue

                if operation_key.lower() == "intensity_shift".lower():
                    for channel, value  in operation_value:
                        image = image_intensity_shift(image,value,channel)
                    continue

                if operation_key.lower() == "new_background".lower():
                    image = new_background_image(image,operation_value["alpha"],np.random.choice(operation_value["paths"],1)[0])       
                    continue

                if operation_key.lower() == "new_background_mask".lower():
                    image = new_background_image_mask(image,image_boxes.copy(),operation_value["alpha"],np.random.choice(operation_value["paths"],1)[0])   
                    continue

            image_data, boxes = process_data_new(image,image_boxes)
                    
            result_images.append(image_data)
            result_boxes.append(boxes)


        # find the max number of boxes
        max_boxes = 0
        for boxz in result_boxes:
            if len(boxz) > max_boxes:
                max_boxes = len(boxz)

        # add zero pad for training
        for i, boxz in enumerate(result_boxes):
            if len(boxz)  < max_boxes:
                zero_padding = np.zeros( (max_boxes-len(boxz), 5), dtype=np.float32)
                result_boxes[i] = np.vstack((boxz, zero_padding))



        result_images = np.array(result_images)
        result_boxes = np.array(result_boxes)

        detectors_mask, matching_true_boxes = get_detector_mask(result_boxes, self.anchors)

        return [result_images, result_boxes, detectors_mask, matching_true_boxes], np.zeros(len(index_array))
    
    def _convert_xml(self, xml_path):
        DOMTree = xml.dom.minidom.parse(xml_path)
        collection = DOMTree.documentElement

        image_path = collection.getElementsByTagName("Datei")[0].getAttribute("Name")
        
        sub_image_rechts = []
        for marker in collection.getElementsByTagName("Marker"):
            if marker.getAttribute("Type") == "S_ROI":
        
                pattern_id = 1
                for label in marker.getElementsByTagName("Label"):
                    # check json for correct name to label conversion. 
                    if label.getAttribute("value") == "pos":
                        pattern_id = 1

                for rect in marker.getElementsByTagName("Rect"):
                    x_min = max(0, int(rect.getAttribute("x")))
                    y_min = max(0, int(rect.getAttribute("y")))

                    x_max = x_min + int(rect.getAttribute("width"))    
                    y_max = y_min + int(rect.getAttribute("height"))
            
                    ## class, x_min, y_min, x_max, y_max
                    sub_image_rechts.append(np.array([pattern_id,x_min,y_min,x_max,y_max]))

        return image_path, np.array(sub_image_rechts)



#http://docs.opencv.org/3.1.0/da/d6e/tutorial_py_geometric_transformations.html
def apply_transform(img,
                    transform_matrix):

    """Apply the image transformation specified by a matrix.
    # Arguments
        x: 2D numpy array, single image.
        transform_matrix: Numpy array specifying the geometric transformation.
    # Returns
        The transformed version of the input.
    """
    rows,cols = img.shape[:2]
    dst = cv2.warpAffine(img,transform_matrix,(cols,rows))


    return dst

def random_shift(img, wrg, hrg, rects, row_axis=0, col_axis=1, channel_axis=2,
                 fill_mode='nearest', cval=0.):
    """Performs a random spatial shift of a Numpy image tensor.
    # Arguments
        x: Input tensor. Must be 3D.
        wrg: Width shift range, as a float fraction of the width.
        hrg: Height shift range, as a float fraction of the height.
        row_axis: Index of axis for rows in the input tensor.
        col_axis: Index of axis for columns in the input tensor.
        channel_axis: Index of axis for channels in the input tensor.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.
    # Returns
        Shifted Numpy image tensor.
    """
    h, w = img.shape[row_axis], img.shape[col_axis]
    tx = np.random.uniform(-hrg, hrg) * h
    ty = np.random.uniform(-wrg, wrg) * w

    min_x = min(rects[:,1])
    max_x = max(rects[:,3])
    min_y = min(rects[:,2])
    max_y = max(rects[:,4])

    if tx + min_x < 0:
        tx = -min_x
    if tx + max_x > w:
        tx = w-max_x
    if ty + min_y < 0:
        ty = -min_y
    if ty + max_x > h:
        ty = h - max_y

    translation_matrix = np.float32([[1,0,tx],[0,1,ty]])

    transform_matrix = translation_matrix  # no need to do offset
    img = apply_transform(img, transform_matrix)

    rects[:,1] = rects[:,1] + tx
    rects[:,3] = rects[:,3] + tx

    rects[:,2] = rects[:,2] + ty
    rects[:,4] = rects[:,4] + ty

    return img, rects

def flip_image(image, rects, axis = 1):
    """
    axis equals zero = vertical flip 
    axis equals one = horizontal flip 
    axis equals minus one = flip both axis
    axis somethink else no flip
    """

    if axis not in [-1,0,1]:
        return image, rects

    height = image.shape[0]
    widht = image.shape[1]
    
    image = cv2.flip(image,axis)

    if axis == 1:
        rects[:,1] = widht - rects[:,1]
        rects[:,3] = widht - rects[:,3]
        rects[:,3], rects[:,1] = rects[:,1].copy(),rects[:,3].copy()
    elif axis == 0:
        rects[:,2] = height - rects[:,2]
        rects[:,4] = height - rects[:,4]
        rects[:,4], rects[:,2] = rects[:,2].copy(),rects[:,4].copy()
    elif axis == -1:
        rects[:,1] = widht - rects[:,1]
        rects[:,3] = widht - rects[:,3]
        rects[:,3], rects[:,1] = rects[:,1].copy(),rects[:,3].copy()
        rects[:,2] = height - rects[:,2]
        rects[:,4] = height - rects[:,4]
        rects[:,4], rects[:,2] = rects[:,2].copy(),rects[:,4].copy()

    return image, rects

def image_intensity_shift(image, intensity, channel = 1, min_intensity = 0, max_intensity = 255):

    non_zero_pixel = np.where(image[:,:,channel] > 0)
    min = np.min(image[:,:,channel][non_zero_pixel])
    max = np.max(image[:,:,channel])

    new_intensity_center = int(np.random.uniform(-intensity, intensity))

    if new_intensity_center + min < 0:
        new_intensity_center = -min
    if new_intensity_center + max > 255:
        new_intensity_center = 255 -  max
        
    image[:,:,channel][non_zero_pixel] = image[:,:,channel][non_zero_pixel] + new_intensity_center

    return image

def new_background_image_mask(image,rects,alpha_value = 0.7, background_image_path = "images/giraffe.jpg"):

    foreground = image.copy().astype(float)
    background = cv2.imread(background_image_path).astype(float)
    background = cv2.resize(background,(foreground.shape[1],foreground.shape[0]))

    alpha = np.zeros_like(foreground).astype(float)   
    
    for rect in rects:
        xmin = rect[1]
        ymin = rect[2]

        width = rect[3] - xmin
        hight = rect[4] - ymin
        cv2.rectangle(alpha,(xmin,ymin),(xmin+width,ymin+hight),(alpha_value,alpha_value,alpha_value),-1)

    foreground = cv2.multiply(alpha, foreground)

    background = cv2.multiply(1 - alpha, background)

    outImage = cv2.add(foreground, background)

    return outImage.astype(int)

def new_background_image(image,alpha_value = 0.7,background_image_path = "images/giraffe.jpg"):

    background = cv2.imread(background_image_path)
    background = cv2.resize(background,(image.shape[1],image.shape[0]))

    result = cv2.addWeighted(image,alpha_value,background,1-alpha_value,0)

    return result



# Args
argparser = argparse.ArgumentParser(
    description="Retrain or 'fine-tune' a pretrained YOLOv2 model for your own data.")

argparser.add_argument(
    '-d',
    '--data_path',
    help="path to the training data",
    default=os.path.join('model_data', 'Hep', 'train_images_yolo.p'))

argparser.add_argument(
    '-v',
    '--validation_data_path',
    help="path to the validation data",
    default=os.path.join('model_data', 'Hep', 'validation_images_yolo.p'))

argparser.add_argument(
    '-a',
    '--anchors_path',
    help='path to anchors file, defaults to yolo_anchors.txt',
    default=os.path.join('model_data',"Hep", 'yolo_anchors.txt'))

argparser.add_argument(
    '-c',
    '--classes_path',
    help='path to classes file, defaults to pascal_classes.txt',
    default=os.path.join('model_data', 'Hep', 'hep_classes.txt'))

# Default anchor boxes
YOLO_ANCHORS = np.array(
    ((0.57273, 0.677385), (1.87446, 2.06253), (3.33843, 5.47434),
     (7.88282, 3.52778), (9.77052, 9.16828)))


def _main(args):
    data_path = os.path.expanduser(args.data_path)
    validation_data_path = os.path.expanduser(args.validation_data_path)
    classes_path = os.path.expanduser(args.classes_path)
    anchors_path = os.path.expanduser(args.anchors_path)

    class_names = get_classes(classes_path)
    anchors = get_anchors(anchors_path)

    xml_paths = glob.glob('C:/Users/Bronzi/Downloads/Train/*.xml', recursive=True)

    random.shuffle(xml_paths)
    
    imageDataGenerator_training = ImageDataGenerator(args.data_path,2,anchors)
    
    #for stuff in imageDataGenerator_training:
    #    break;
    #    print ("")
    
    #imageDataGenerator_validation = ImageDataGenerator(args.data_path,2,anchors)

    operations = {
                    "flip":[1,0,-1,4],
                    "random_shift":
                            {
                                "max_x_shift":0.5,
                                "max_y_shift":0.5
                            },
                    "intensity_shift": [(1,50),(2,50)],
                }

    imageDataGenerator_training = ImageDataGeneratorXML_RegionCNN(xml_paths[:30],2,anchors,operations)
    imageDataGenerator_validation = ImageDataGeneratorXML_RegionCNN(xml_paths[30:] ,2,anchors)

    anchors = YOLO_ANCHORS
    model_body, model = create_model(anchors, class_names)

    train(
        model,
        class_names,
        anchors,
        imageDataGenerator_training,
        imageDataGenerator_validation
    )

    #draw(model_body,
    #    class_names,
    #    anchors,
    #    imageDataGenerator_validation,
    #    weights_name='trained_stage_3_best.h5',
    #    save_all=False)


def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    if os.path.isfile(anchors_path):
        with open(anchors_path) as f:
            anchors = f.readline()
            anchors = [float(x) for x in anchors.split(',')]
            return np.array(anchors).reshape(-1, 2)
    else:
        Warning("Could not open anchors file, using default.")
        return YOLO_ANCHORS

def process_data_new(image, boxes=None):
    '''processes the data'''
    #images = [Image.open(i).convert('RGB') for i in images]
    orig_size = np.array([image.shape[1], image.shape[0]]) # width, height
    orig_size = np.expand_dims(orig_size, axis=0) 

    # Image preprocessing.
    processed_image = cv2.resize(image,(target_width, target_height)).astype(float)
    processed_image = processed_image / 255.

    #processed_images = [i.resize((target_width, target_height), Image.BICUBIC) for i in images]
    #processed_images = [np.array(image, dtype=np.float) for image in processed_images]
    #processed_images = [image/255. for image in processed_images]

    if boxes is not None:
        # Box preprocessing.
        # Original boxes stored as 1D list of class, x_min, y_min, x_max, y_max.
        boxes = [box.reshape((-1, 5)) for box in boxes]
        # Get extents as y_min, x_min, y_max, x_max, class for comparision with
        # model output.
        boxes_extents = [box[:, [2, 1, 4, 3, 0]] for box in boxes]

        # Get box parameters as x_center, y_center, box_width, box_height, class.
        boxes_xy = [0.5 * (box[:, 3:5] + box[:, 1:3]) for box in boxes]
        boxes_wh = [box[:, 3:5] - box[:, 1:3] for box in boxes]
        boxes_xy = [boxxy / orig_size for boxxy in boxes_xy]
        boxes_wh = [boxwh / orig_size for boxwh in boxes_wh]
        boxes = [np.concatenate((boxes_xy[i], boxes_wh[i], box[:, 0:1]), axis=1) for i, box in enumerate(boxes)]

        result = []
        for box in boxes:
            result.append(box.flatten())

        return processed_image, np.array(result)
    else:
        return processed_image

def process_data(images, boxes=None):
    '''processes the data'''
    images = [Image.open(i).convert('RGB') for i in images]
    orig_size = np.array([images[0].width, images[0].height])
    orig_size = np.expand_dims(orig_size, axis=0)

    # Image preprocessing.
    processed_images = [i.resize((target_width, target_height), Image.BICUBIC) for i in images]
    processed_images = [np.array(image, dtype=np.float) for image in processed_images]
    processed_images = [image/255. for image in processed_images]

    if boxes is not None:
        # Box preprocessing.
        # Original boxes stored as 1D list of class, x_min, y_min, x_max, y_max.
        boxes = [box.reshape((-1, 5)) for box in boxes]
        # Get extents as y_min, x_min, y_max, x_max, class for comparision with
        # model output.
        boxes_extents = [box[:, [2, 1, 4, 3, 0]] for box in boxes]

        # Get box parameters as x_center, y_center, box_width, box_height, class.
        boxes_xy = [0.5 * (box[:, 3:5] + box[:, 1:3]) for box in boxes]
        boxes_wh = [box[:, 3:5] - box[:, 1:3] for box in boxes]
        boxes_xy = [boxxy / orig_size for boxxy in boxes_xy]
        boxes_wh = [boxwh / orig_size for boxwh in boxes_wh]
        boxes = [np.concatenate((boxes_xy[i], boxes_wh[i], box[:, 0:1]), axis=1) for i, box in enumerate(boxes)]

        # find the max number of boxes
        max_boxes = 0
        for boxz in boxes:
            if boxz.shape[0] > max_boxes:
                max_boxes = boxz.shape[0]

        # add zero pad for training
        for i, boxz in enumerate(boxes):
            if boxz.shape[0]  < max_boxes:
                zero_padding = np.zeros( (max_boxes-boxz.shape[0], 5), dtype=np.float32)
                boxes[i] = np.vstack((boxz, zero_padding))

        return np.array(processed_images), np.array(boxes)
    else:
        return np.array(processed_images)


def get_detector_mask(boxes, anchors):
    '''
    Precompute detectors_mask and matching_true_boxes for training.
    Detectors mask is 1 for each spatial position in the final conv layer and
    anchor that should be active for the given boxes and 0 otherwise.
    Matching true boxes gives the regression targets for the ground truth box
    that caused a detector to be active or 0 otherwise.
    '''
    detectors_mask = [0 for i in range(len(boxes))]
    matching_true_boxes = [0 for i in range(len(boxes))]
    for i, box in enumerate(boxes):
        detectors_mask[i], matching_true_boxes[i] = preprocess_true_boxes(box, anchors, [target_width, target_height])

    return np.array(detectors_mask), np.array(matching_true_boxes)

def create_model(anchors, class_names, load_pretrained=True, freeze_body=True):
    '''
    returns the body of the model and the model

    # Params:

    load_pretrained: whether or not to load the pretrained model or initialize all weights

    freeze_body: whether or not to freeze all weights except for the last layer's

    # Returns:

    model_body: YOLOv2 with new output layer

    model: YOLOv2 with custom loss Lambda layer

    '''

    detectors_mask_shape = (26, 26, 5, 1)
    matching_boxes_shape = (26, 26, 5, 5)

    # Create model input layers.
    image_input = Input(shape=(target_width, target_height, 3))
    boxes_input = Input(shape=(None, 5))
    detectors_mask_input = Input(shape=detectors_mask_shape)
    matching_boxes_input = Input(shape=matching_boxes_shape)

    # Create model body.
    yolo_model = yolo_body(image_input, len(anchors), len(class_names))
    topless_yolo = Model(yolo_model.input, yolo_model.layers[-2].output)

    if load_pretrained:
        # Save topless yolo:
        topless_yolo_path = os.path.join('model_data', 'yolo_topless.h5')
        if not os.path.exists(topless_yolo_path):
            print("CREATING TOPLESS WEIGHTS FILE")
            yolo_path = os.path.join('model_data', 'yolo.h5')
            model_body = load_model(yolo_path)
            model_body = Model(model_body.inputs, model_body.layers[-2].output)
            model_body.save_weights(topless_yolo_path)
        topless_yolo.load_weights(topless_yolo_path)

    if freeze_body:
        for layer in topless_yolo.layers:
            layer.trainable = False
    final_layer = Conv2D(len(anchors)*(5+len(class_names)), (1, 1), activation='linear')(topless_yolo.output)

    model_body = Model(image_input, final_layer)

    # Place model loss on CPU to reduce GPU memory usage.
    with tf.device('/cpu:0'):
        # TODO: Replace Lambda with custom Keras layer for loss.
        model_loss = Lambda(
            yolo_loss,
            output_shape=(1, ),
            name='yolo_loss',
            arguments={'anchors': anchors,
                       'num_classes': len(class_names)})([
                           model_body.output, boxes_input,
                           detectors_mask_input, matching_boxes_input
                       ])

    model = Model(
        [model_body.input, boxes_input, detectors_mask_input,
         matching_boxes_input], model_loss)

    return model_body, model

def train(model, class_names, anchors, imageDataGenerator_training, imageDataGenerator_validation):
    '''
    retrain/fine-tune the model

    logs training with tensorboard

    saves training weights in current directory

    best weights according to val_loss is saved as trained_stage_3_best.h5
    '''
    model.compile(
        optimizer='adam', loss={
            'yolo_loss': lambda y_true, y_pred: y_pred
        })  # This is a hack to use the custom loss function in the last layer.


    logging_1 = TensorBoard(log_dir="log/log_1")
    logging_2 = TensorBoard(log_dir="log/log_2")
    logging_3 = TensorBoard(log_dir="log/log_3")
    checkpoint = ModelCheckpoint("trained_stage_3_best.h5", monitor='val_loss',
                                 save_weights_only=True, save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=15, verbose=1, mode='auto')


    model.fit_generator(imageDataGenerator_training,
                         validation_data= imageDataGenerator_validation,
                         samples_per_epoch = 30,
                         callbacks=[logging_1],
                         nb_epoch=10,
                         validation_steps=1)

    #model.fit([image_data, boxes, detectors_mask, matching_true_boxes],
    #          np.zeros(len(image_data)),
    #          validation_split=validation_split,
    #          batch_size=32,
    #          epochs=100,
    #          callbacks=[logging])
    model.save_weights('trained_stage_1.h5')

    model_body, model = create_model(anchors, class_names, load_pretrained=False, freeze_body=False)

    model.load_weights('trained_stage_1.h5')

    model.compile(
        optimizer='adam', loss={
            'yolo_loss': lambda y_true, y_pred: y_pred
        })  # This is a hack to use the custom loss function in the last layer.


    model.fit_generator(imageDataGenerator_training,
                        validation_data= imageDataGenerator_validation,
                        samples_per_epoch = 30,
                        callbacks=[logging_2],
                        nb_epoch=10,
                        validation_steps=1)

    #model.fit([image_data, boxes, detectors_mask, matching_true_boxes],
    #          np.zeros(len(image_data)),
    #          validation_split=0.1,
    #          batch_size=8,
    #          epochs=4*30,
    #          callbacks=[logging])

    model.save_weights('trained_stage_2.h5')

    #model.fit([image_data, boxes, detectors_mask, matching_true_boxes],
    #          np.zeros(len(image_data)),
    #          validation_split=0.1,
    #          batch_size=8,
    #          epochs=4*30,
    #          callbacks=[logging, checkpoint, early_stopping])

    model.fit_generator(imageDataGenerator_training,
                    validation_data= imageDataGenerator_validation,
                    samples_per_epoch = 30,
                    callbacks=[logging_3,checkpoint],
                    nb_epoch=10,
                    validation_steps=1)

    model.save_weights('trained_stage_3.h5')

    #model.save("YOLO_Hep.hdf5")

def draw(model_body, class_names, anchors, image_data, image_set='val',
            weights_name='trained_stage_3_best.h5', out_path="output_images", save_all=True):
    '''
    Draw bounding boxes on image data
    '''

    # model.load_weights(weights_name)
    #print(image_data.shape)
    model_body.load_weights(weights_name)

    # Create output variables for prediction.
    yolo_outputs = yolo_head(model_body.output, anchors, len(class_names))
    input_image_shape = K.placeholder(shape=(2, ))
    boxes, scores, classes = yolo_eval(
        yolo_outputs, input_image_shape, score_threshold=0.07, iou_threshold=0)

    # Run prediction on overfit image.
    sess = K.get_session()  # TODO: Remove dependence on Tensorflow session.

    if  not os.path.exists(out_path):
        os.makedirs(out_path)
    for i in range(len(image_data)):
        out_boxes, out_scores, out_classes = sess.run(
            [boxes, scores, classes],
            feed_dict={
                model_body.input: image_data[i],
                input_image_shape: [image_data.shape[2], image_data.shape[3]],
                K.learning_phase(): 0
            })
        print('Found {} boxes for image.'.format(len(out_boxes)))
        print(out_boxes)

        # Plot image with predicted boxes.
        image_with_boxes = draw_boxes(image_data[i][0], out_boxes, out_classes,
                                    class_names, out_scores)
        # Save the image:
        if save_all or (len(out_boxes) > 0):
            image = Image.fromarray(image_with_boxes)
            image.save(os.path.join(out_path,str(i)+'.png'))

        # To display (pauses the program):
        # plt.imshow(image_with_boxes, interpolation='nearest')
        # plt.show()



if __name__ == '__main__':
    args = argparser.parse_args()
    _main(args)
