import h5py
import numpy as np
import os
import cv2
from tqdm import tqdm
import pickle

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

target_width = 416
target_height = 416
batch_size = 1

image_paths = []
boxes = []


image_counter = 0
for image, mask, pattern_id in tqdm(zip(images, images_masks, pattern_ids)):
    target_subimages_count = 1 + ((max_count / id_counts_dict[pattern_id]) - 1) # 75 + 20 * 

    for i in range(0,int(target_subimages_count),batch_size):

        height = image.shape[0]
        width = image.shape[1]

        x_start = np.random.randint(0,width-target_width)
        y_start = np.random.randint(0,height-target_height)
        
        sub_image = image[y_start:y_start+target_height,x_start:x_start+target_width]
        sub_mask = mask[y_start:y_start+target_height,x_start:x_start+target_width]

        _, cnts, _ = cv2.findContours(sub_mask.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


        sub_image_rechts = []
        for c in cnts:

            (x, y, w, h) = cv2.boundingRect(c)

            if x > 0 and y > 0 and x + w < target_width and y + h < target_height:

                # class, x_min, y_min, x_max, y_max
                sub_image_rechts.append(np.array([pattern_id,x,y,x+w,y+h]))

        if len(sub_image_rechts) > 0:
            sub_image_rechts  = np.array(sub_image_rechts)

            boxes.append(sub_image_rechts) 

            sub_image_path = "Images/Hep/{0:07d}.png".format(image_counter)
            cv2.imwrite(sub_image_path,sub_image)
            image_paths.append(sub_image_path)

            image_counter += 1             


train_augmented_file_path = "./train_images_yolo.p" 


with open(train_augmented_file_path, 'wb') as train_augmented_file:
    output_dict = {}
    output_dict.update({"images":image_paths})
    output_dict.update({"boxes":np.array(boxes)})
    pickle.dump(output_dict, train_augmented_file)




