# Shape Appearance Units

Implement the shape appearance selectivity unit as described in the paper "Convolutional neural networks explain tuning
properties of anterior, but not middle, face-processing areas in macaque inferotemporal cortex"

For this part, you will need to get the facial landmarks which requires to contact Prof. Haruo Hosoya and accept
his terms. Once you have the landmarks, you can run the script "create_FEI_SA_images" within the "dataset_utils" folder.


**test 1 - Shape/Appearance example Fig. 4a **

<img src='../img/shape-3.jpg' height="660"><img src='../img/shape0.jpg' height="660"><img src='../img/shape3.jpg' height="660">
1st shape dimension

<img src='../img/appareance-3.jpg' height="660"><img src='../img/appareance0.jpg' height="660"><img src='../img/appareance3.jpg' height="660">
1st appearance dimension


**test 2 - AM/ ML distribution on VGG19 trained on imagenet **
<img src='../img/SA_unit.png' height="660">
SPI index distribution of the face units for VGG19

config used:
{
  "model": "VGG19",
  "include_top": true,
  "weights": "imagenet",
  "train_data": "FEI_SA",
  "lmk_path":"<...data/FEI/landmark_files>",
  "orig_img_path":"<...>data/FEI/face_images/originalimages>",
  "SA_img_path": "<...data/FEI/face_images/shape_appearance_images>",
  "front_view": "view_11.mat",
  "csv": "<...>data/FEI/FEI_SA_features.csv>",
  "face_units_path": "models/saved/face_units",
  "save_path": "models/saved/SA_units"
}