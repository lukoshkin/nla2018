# NLA project (2018)
---
In this mini-project, we have managed to substitute unstructured matrices in fully-connected layers with block-circulant matrices. This has been performed on a CNN, designed for addressing image classification problem (on the well-studied dataset CIFAR10). As it supposed, a CirCNN (the name, as well as the main idea of this project, comes from [this article](https://arxiv.org/pdf/1708.08917.pdf)) should outperform regular CNNs, which is confirmed by the authors. Our implementation certainly wins at the amount of memory occupied, however, the acceleration in operation does not occur, which is probably due to imperfect coding style. The next step is to implement the block-circulant structure in the weight matrices of the convolutional layers. In the article there are some uncertainties in the section devoted to the convolutional layer. It is good to start with this if it is to be continued.  
You can compare the performance of our networks with regular ones which are given [here](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py).
