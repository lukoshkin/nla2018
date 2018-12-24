# nla_project2018
Team #9
***

In this mini-project we have managed to substitute unstructured matrices in fully-connected layers with block-circulant matrices. This has been performed on a CNN, designed for adressing image classification problem (on the well-studied dataset CIFAR10).  
As it supposed, a CirCNN (the name as well as the main idea of this project comes from [this article](https://arxiv.org/pdf/1708.08917.pdf))
should outperform regular CNNs, which is confirmed by the authors. Our implementation certainly wins at the amount of memory occupied,
however the acceleration in operation does not occur, which is probably due to imperfect codding style. Since it is open-source code,  
everyone is wellcome to сontribute to the development of the project. Also, we are going to move further and complete the work with implementing the block-circulant structure in weight matrices of the convolutional layers. Here the assistance may be required as well
(in article there are some uncertainties in the section devoted to the convolutional layer)
