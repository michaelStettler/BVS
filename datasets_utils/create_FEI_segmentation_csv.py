import os
import pandas as pd
import numpy as np
import glob
import cv2
import matplotlib.pyplot as plt

# path = 'D:/Dataset/FEI'  # windows
path = '/Users/michaelstettler/PycharmProjects/BVS/data/FEI'  # personal mac
seg_path = 'face_images/segmentedimages'
save_label_path = 'seg_labels'
csv_name = 'FEI_face_segmented.csv'

# define colors
colors = {
    # "background": [255, 255, 255],
    "Hair": [108, 54, 0],
    "Left_Eye_brow": [175, 88, 0],
    "Right_eye_brow": [162, 81, 0],
    "Left_eye_lid": [240, 255, 0],
    "Right_eye_lid": [240, 255, 40],
    "Eye_pupil": [0,0,0],
    "Eye_white": [255, 227, 199],
    "Nose": [0, 0, 255],
    "Mouth": [255, 0, 0],
    "Left_ear": [0,255, 0],
    "Right_ear": [30, 255, 30],
    "Teeth": [255, 130, 130]
}
num_classes = len(colors)
print("num_classes", num_classes)
threshold = [8, 8, 8]
# get all segmented files
seg_images_list = glob.glob(os.path.join(path, seg_path) + '/*.png')

# declare dataframe
df = pd.DataFrame(columns=('img', 'seg_img', 'label'))

# load each images
for image_path in seg_images_list:
    im_name = image_path.split('/')[-1]
    print("image name:", im_name)

    # load image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # plt.figure()
    # plt.imshow(img)

    seg_mask = np.zeros((np.shape(img)[0], np.shape(img)[1], num_classes), dtype=np.uint8)

    # segment each part by its color
    for c, color in enumerate(colors):
        # build lower and upper boundaries conditions as the formating colors are a bit off form time to time
        low_cond = np.maximum(np.array(colors[color]) - threshold, [0, 0, 0])
        up_cond = np.minimum(np.array(colors[color]) + threshold, [255, 255, 255])
        # declare segmentation mask
        seg = np.zeros(np.shape(img))
        # red channel
        seg_r = seg[:, :, 0]
        img_r = img[:, :, 0]
        seg_r[img_r >= low_cond[0]] = 1
        seg_r[img_r > up_cond[0]] = 0
        seg[:, :, 0] = seg_r
        # green channel
        seg_g = seg[:, :, 1]
        img_g = img[:, :, 1]
        seg_g[img_g >= low_cond[1]] = 1
        seg_g[img_g > up_cond[1]] = 0
        seg[:, :, 1] = seg_g
        # blue channel
        seg_b = seg[:, :, 2]
        img_b = img[:, :, 2]
        seg_b[img_b >= low_cond[2]] = 1
        seg_b[img_b > up_cond[2]] = 0
        seg[:, :, 2] = seg_b

        # get only the positions where all conditions agree
        seg = np.prod(seg, axis=2)

        # plt.figure()
        # plt.imshow(seg)
        # plt.title(color)

        if np.sum(seg) < 1:
            print("Warning nothing found for label", color)

        seg_mask[:, :, c] = seg

    # add entry to dataframe
    im_nam = im_name.split('.')[0]
    df = df.append({'img': im_nam + '.jpg', 'seg_img': im_name, 'label': im_nam + '.npy'}, ignore_index=True)

    # save the segmented tensor
    np.save(os.path.join(path, save_label_path) + '/' + im_name.split('.')[0], seg_mask)

# save csv
df.to_csv(os.path.join(path, csv_name), index=False)

# plt.show()
