<h1> BVS </h1>
Brain Visual System modelling toolbox

<h2>Introduction</h2>
The toolbox aim to model different visual pathway of the human brain. 
The toolbox is built to help putting different submodules as to try different 
theoretical approaches and study their implication as a system on different task; 
i.e. categorization, classification etc. 

The motivation lies since many early work in machine learning and computer
vision were focused on building specific features but has been faded away since
deep learning methods shows a fantastic increase of performance. Yet systems
using deep learning methods have become treated more and more as black box, in such, 
we lost in explanability of the components while favoring a general 
performance score. With this toolbox, I hope to leverage the power and 
scalability of common deep learning framework such as TensorFlow to implement 
basic theoretical component of our visual system and allow to merge/combine 
these components with state of the art architecture and study more in depth 
their implication. 

This repository contains so far three physiologically inspired models, an early V1 
system that aims to compute a saliency map, a second model for the recognition
of body interaction and finally a model that explain tuning for facial identity. 

The toolbox is currently aiming at building models to represent how the brain 
perceive dynamic facial expressions. The reason lies as many dataset are
available to train model on faces, which can be compared with many psychological 
and behavioral studies.

Moreover, this repository contain many useful implementation of different paper to help
build, generate and visualize activation at different level of the models. 

<h2>What to find</h2>
In this repository, you will find:

<h3>A complete implementation of the V1 model and its bottom-up saliency map </h3>
as a TensorFlow layer following the description of the book (bvs.layers.BotUpSaliency.py):

>Zhaoping, L., & Li, Z. (2014). Understanding vision: theory, models, and data. Oxford University Press, USA.

See reports: "BottomUpSaliency_layer" for main results on this part.

<h3>An implementation of the norm base mechanism </h3>
(models.NormBase.py): 

>Giese, M. A., & Leopold, D. A. (2005). Physiologically inspired neural model for the encoding of face spaces. Neurocomputing, 65, 93-101

>Stettler, M., Taubert, N., Azizpour, T., Siebert, R., Spadacenta, S., Dicke, P., ... & Giese, M. A. (2020, September). Physiologically-Inspired Neural Circuits for the Recognition of Dynamic Faces. In International Conference on Artificial Neural Networks (pp. 168-179). Springer, Cham.

<h3>A Face Selective Index (FSI) and Shape Preference Index (SPI) unit extraction </h3> 
as explain in the methods of (utils.find_face_units.py):

>Raman, R., & Hosoya, H. (2020). Convolutional neural networks explain tuning properties of anterior, but not middle, face-processing areas in macaque inferotemporal cortex. Communications biology, 3(1), 1-14.
 
See reports: "Face_units" and "ShapeAppearance_units" for main results. 
 
<h3>A semantic concept face-parts score </h3>
which is a derivation of the work from (utils.find_semantic_units.py):

> Zhou, B., Oliva, A., & Torralba, A. (2018). Network Dissection: Quantifying Interpretability of Deep Visual Representation.

See reports: "FaceParts_Semantic_Units".

Where I implemented the score for segmentation concept as a data-set-wide intersection over union (IoU) for 12 facial concepts, 
where I have manually segmented (on going work) the facial parts on the FEI dataset (frontal pose). 

<h2>Acknowledgments</h2>
I would like to thanks Prof. Zhaoping Li for her very valuable inputs on her 
model and the very interesting discussion about what it means to have a V1 
area responsible to compute a bottom-up saliency map.

I am also thankful to Rajani Raman and Haruo Hosoya for their fruitful answers to my questions and to have accepted to 
share their data with me. 
