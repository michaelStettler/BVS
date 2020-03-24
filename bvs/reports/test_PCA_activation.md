# PCA activation over sequence

script: t07_test_PCA_activation.py

config file: config_test3.json

**test 1 - 10 components activation PCA:**
Used only 1 layer with 3 gabor filters types (config_test3.json). Kept 10 principal component,

explained variance: [0.693 0.141 0.0527 0.027 0.017 0.008 0.008 0.004 0.003 0.002]

![](../../img/PCA_10c.gif)

heatmap of PCA activation over the sequence

**test 1 - 50 components activation PCA:**
Used only 1 layer with 3 gabor filters types (config_test3.json). Kept 50 principal component,

explained variance: 
[6.9326234e-01 1.4175528e-01 5.2785397e-02 2.7328748e-02 1.7694645e-02
 8.7410305e-03 8.1469156e-03 4.4940682e-03 3.4616753e-03 2.7224983e-03
 2.4835134e-03 1.7045854e-03 1.4374203e-03 1.3604553e-03 1.1369602e-03
 9.0654427e-04 8.3563157e-04 7.1078539e-04 6.5139087e-04 5.5160763e-04
 4.9644202e-04 4.4435915e-04 3.8599220e-04 3.7223060e-04 3.4849579e-04
 3.0378031e-04 2.8337154e-04 2.5932648e-04 2.4479747e-04 2.2932155e-04
 2.2397311e-04 2.1764902e-04 2.0883238e-04 2.0338918e-04 2.0192868e-04
 1.9937297e-04 1.9526934e-04 1.9272882e-04 1.8986351e-04 1.8750635e-04
 1.8615110e-04 1.8205361e-04 1.8060241e-04 1.7965880e-04 1.7709934e-04
 1.7651166e-04 1.7105186e-04 1.6955007e-04 1.6706037e-04 1.6635279e-04]

![](../../img/PCA_50c.gif)

heatmap of PCA activation over the sequence
