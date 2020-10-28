<h1> BVS </h1>
Brain Visual System modelling toolbox

<h2>Introduction</h2>
The toolbox aim to model the different visual pathway of the human brain. 
The toolbox is built to help putting different submodules as to try different 
theoretical approaches and study their implication as a system on different task; 
i.e. categorization, classification etc. 

The motivation lies since many early work in machine learning and computer
vision were focused on building specific features but has been faded since
deep learning methods shows a fantastic increase of performance. Yet systems
using deep learning methods have become treated as black box, in such, 
we lost in expandability of the components while favoring a general 
performance score. With this toolbox, I hope to leverage the power and 
scalability of common deep learning framework such as TensorFlow to implement 
basic theoretical component of our visual system and allow to merge/combine 
these components with state of the art architecture and study more in depth 
their implication. 

While many research focus on studying the similarity of Neuronal Networks 
compared to behavioural study. Where Prof. DiCarlo even proposed a brain score 
to quantify Deep learning architectures. With this work, I hope to center the 
discussion back to a theoretical approaches were one could start with an 
hypothesis and investigate its assumptions as a system. By using common 
framework, one could study how his/her hypothesis impacts with the overall 
performance score, but also keeping in mind the complexity of the model and 
especially the explainability of each of its components. Of course there's 
no free lunch, and such models add lots of induces biased which will hardly
compete with state of the art end-to-end deep learning architectures. But, 
understanding what components may affect the system could shed light to better
end-to-end system. 

In such, I would like to follow the idea of a "brain score" mechanism but 
related to each modules. I hope with this work to help researchers to focus 
on the brain area which are less understood and propose mechanism to build 
more robust system.

The toolbox will first focus on building models to represent how the brain 
perceive dynamic facial expressions. The reason lies as many dataset are
available to train model on faces, and many psychological and behavioral 
study will draw the general line to compute the a "BVS" scores. 

<h2>What to find</h2>
In this repository, you will find a complete implementation of the V1 model and 
its bottom-up saliency map describe in the book:

>Zhaoping, L., & Li, Z. (2014). Understanding vision: theory, models, and data. Oxford University Press, USA.

And, you will also find an implementation of the norm base mechanism: 

> Giese, M. A., & Leopold, D. A. (2005). Physiologically inspired neural model for the encoding of face spaces. Neurocomputing, 65, 93-101

> Stettler, M., Taubert, N., Azizpour, T., Siebert, R., Spadacenta, S., Dicke, P., ... & Giese, M. A. (2020, September). Physiologically-Inspired Neural Circuits for the Recognition of Dynamic Faces. In International Conference on Artificial Neural Networks (pp. 168-179). Springer, Cham.

<h2>Acknowledgment</h2>
I would like to thanks Prof. Zhaoping Li for her very valuable inputs on her 
model and the very interesting discussion about what it means to have a V1 
area responsible to compute a bottom-up saliency map.  
