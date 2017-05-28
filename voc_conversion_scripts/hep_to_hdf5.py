import h5py
import numpy as np
import os
import cv2
from tqdm import tqdm
import pickle
import imutils

from sklearn.model_selection import train_test_split

def random_flip(image, mask):
    # a flag to specify how to flip the array; 0 means flipping around the x-axis and positive value (for example, 1) means flipping around y-axis. Negative value (for example, -1) means flipping around both axes 4 means no flipping       
    flipCode = np.random.randint(-1,4)

    if flipCode == 4:
        return image, mask
    
    image = cv2.flip(image,flipCode)
    mask = cv2.flip(mask,flipCode)

    return image,mask

def rotate_new_size(image,mask):

    angle = np.random.randint(0,360)

    image = imutils.rotate_bound(image, angle)
    mask = imutils.rotate_bound(mask, angle)

    return image,mask

def rotate(image,mask):

    angle = np.random.randint(0,360)

    image = imutils.rotate(image, angle)
    mask = imutils.rotate(mask, angle)

    return image,mask


train_augmented_file_path = "E:/Cloud/train_images_v3.p" 

with open(train_augmented_file_path, mode='rb') as f:
    train = pickle.load(f)


images, images_masks, pattern_ids = train['features'],train['labels'],train['pattern_id']


print(len(images))
print(len(images_masks))
print(len(pattern_ids))

ids, index, counts = np.unique(pattern_ids, return_index = True, return_counts=True)
id_counts_dict = dict(zip(ids,counts))
max_count = max(counts)

target_width = 416*2
target_height = 416*2
batch_size = 1

image_paths_train = []
boxes_train = []

image_paths_validation = []
boxes_validation = []


image_counter = 0
for image, mask, pattern_id in tqdm(zip(images, images_masks, pattern_ids)):
    target_subimages_count = 5 + 10 * ((max_count / id_counts_dict[pattern_id]) - 1) # 75 + 20 * 

    is_train_image = np.random.choice([True,False], 1, p=[0.8, 0.2])[0]

    for i in range(0,int(target_subimages_count),batch_size):

        height = image.shape[0]
        width = image.shape[1]

        temp_image, temp_mask = random_flip(image, mask)
        temp_image, temp_mask = rotate(temp_image, temp_mask)

        x_start = np.random.randint(0,width-target_width)
        y_start = np.random.randint(0,height-target_height)
        
        sub_image = temp_image[y_start:y_start+target_height,x_start:x_start+target_width]
        sub_mask = temp_mask[y_start:y_start+target_height,x_start:x_start+target_width]

        _, cnts, _ = cv2.findContours(sub_mask.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


        sub_image_rechts = []
        for c in cnts:

            (x, y, w, h) = cv2.boundingRect(c)

            if x > 0 and y > 0 and x + w < target_width and y + h < target_height:

                # class, x_min, y_min, x_max, y_max
                sub_image_rechts.append(np.array([pattern_id,x,y,x+w,y+h]))

        if len(sub_image_rechts) > 0:
            sub_image_rechts  = np.array(sub_image_rechts)

            if (is_train_image):

                boxes_train.append(sub_image_rechts) 

                sub_image_path = "Images/Hep/Training/{0:07d}.png".format(image_counter)
                cv2.imwrite(sub_image_path,sub_image)
                image_paths_train.append(sub_image_path)

            else:
                boxes_validation.append(sub_image_rechts) 

                sub_image_path = "Images/Hep/Validation/{0:07d}.png".format(image_counter)
                cv2.imwrite(sub_image_path,sub_image)
                image_paths_validation.append(sub_image_path)


            image_counter += 1             

            

with open(os.path.join('model_data', 'Hep', 'train_images_yolo.p') , 'wb') as train_augmented_file:
    output_dict = {}
    output_dict.update({"images":image_paths_train})
    output_dict.update({"boxes":np.array(boxes_train)})
    pickle.dump(output_dict, train_augmented_file)

with open(os.path.join('model_data', 'Hep', 'validation_images_yolo.p') , 'wb') as validation_augmented_file:
    output_dict = {}
    output_dict.update({"images":image_paths_validation})
    output_dict.update({"boxes":np.array(boxes_validation)})
    pickle.dump(output_dict, validation_augmented_file)


