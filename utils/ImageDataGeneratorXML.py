import cv2
import numpy as np
import os

from xml.dom.minidom import parse
import xml.dom.minidom
from tqdm import tqdm


operations = {
                "flip":[1,0,-1,4],
                "random_shift":
                        {
                            "max_x_shift":0.5,
                            "max_y_shift":0.5
                        },
                "intensity_shift": [(1,50),(2,50)],
                #"new_background" : 
                #        {
                #            "paths" : ["images/giraffe.jpg","images/horses.jpg"],
                #            "alpha" : 0.7                                                
                #        },
                    #"new_background_mask" : 
                    #    {
                    #        "paths" : ["images/giraffe.jpg","images/horses.jpg"],
                    #        "alpha" : 0.7                                                
                    #    }

                }

print ("")

from keras.preprocessing.image import Iterator

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
                    
            result_images.append(image.astype(np.uint8))
            result_boxes.append(image_boxes)

        return result_images,result_boxes
    
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

def draw_results(image, rects):

    for rect in rects:
        xmin = rect[1]
        ymin = rect[2]

        width = rect[3] - xmin
        hight = rect[4] - ymin
        cv2.rectangle(image,(xmin,ymin),(xmin+width,ymin+hight),(255,0,0),15)

    return image


xml_paths = []

basic_path = "C:/Users/Bronzi/Downloads/Train/"
xml_paths.append(os.path.join(basic_path,"20170602-Linien-Array-F170530SA_19_100_G_Test_stitched__Rack-1-OT-2-Col-4-Row-2-Chip-1_eMarker.xml"))
xml_paths.append(os.path.join(basic_path,"20170428-ENA-Test_1_100_G_Test_stitched__Rack-1-OT-1-Col-1-Row-1-Chip-1_eMarker.xml"))

imageDataGeneratorXML_RegionCNN = ImageDataGeneratorXML_RegionCNN(xml_paths,1,None,operations)

for images, boxes in imageDataGeneratorXML_RegionCNN:
    
    for image, box in zip(images,boxes):

        result_image = draw_results(image,box)

        cv2.namedWindow('image', cv2.WINDOW_FREERATIO)
        cv2.imshow('image',result_image)
        cv2.waitKey(0)

print ("test")

