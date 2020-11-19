# Face Units

Implement the face selectivity unit as described in the paper "Convolutional neural networks explain tuning properties
of anterior, but not middle, face-processing areas in macaque inferotemporal cortex"

** Procedure **
The FEI dataset can be obtain here: https://fei.edu.br/~cet/facedatabase.html

For the non-face images, I randomly picked 50 images from the "imagenet" dataset.
Then you will need to run the script "create_FEI_csv.py" within the "datasets_utils" folder. Make sure to modify the
configuration file to fit your data path folder.

**test 1 - VGG19 result**

<img src='../img/face_unit_VGG19.png' height="660">

config used:

{
  "model": "VGG19",
  "include_top": true,
  "weights": "imagenet",
  "train_data": "FEI",
  "orig_img_path":"<...data/FEI/face_images/originalimages>",
  "front_view": "view_11.mat",
  "csv": "<...>/data/FEI/FEI_face_units.csv>",
  "save_path": "models/saved/face_units"
}