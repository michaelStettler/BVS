# Max Pooling Depths report results

script t03_test_visualize_max_pool_activation.py
config file: config_test3.json

**test1 - 2D max pool**
ksize=(3, 3), strides=1, padding='SAME'

<img src='../../img/heatmap_MaxPooling_per_channel.jpeg' height="330">
heatmap of max pooling activation on 3 different gabor filter type (sigma=2, lambda=1), (sigma=3, lambda=2), (sigma=4, lamda=3)

**test2 - depths max pool**
axis=3, num_cond=3 (ksize=(1, 1), strides=1, padding='SAME')

<img src='../../img/heatmap_MaxPooling_only_depths.jpeg' height="110">
heatmap of depth max pooling activation on 3 different gabor filter type (sigma=2, lambda=1), (sigma=3, lambda=2), (sigma=4, lamda=3)

**test2 - depths and 2D max pool**
axis=3, num_cond=3 (ksize=(1, 1), strides=1, padding='SAME')

<img src='../../img/heatmap_MaxPooling_depths.jpeg' height="110">
heatmap of depth max pooling activation on 3 different gabor filter type (sigma=2, lambda=1), (sigma=3, lambda=2), (sigma=4, lamda=3)
