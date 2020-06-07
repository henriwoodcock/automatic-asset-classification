# Semi-Automatic Asset Classification
Image classification and object identification with neural networks has been proven very effect. In the 2017 ImageNet Competition <sup>[1](#imageNetFootNote)</sup> 29 of the 38 teams competing achieved an accuracy of greater than 95% [1][2]

Semi-Automatic Asset Classification aims to apply these top standards to asset classification. This is done by training a neural network on a set of _labelled images_. The issue with this type of modelling is a large dataset of already labelled images is required, this can be costly and timely to create. To reach the top accuracy it is argued that more data is more important than better modelling [3], meaning that to keep improving models more data is required. Here, _transfer learning_ can be used to reduce this need or such large datasets.

## Transfer Learning
_Transfer learning_ is the use of applying one neural network to a new problem, the hope is that the learned architecture can be applied to a new problem. This technique has been proven efficient, especially in image classification where pretrained ImageNet models are transferred and fine-tuned to new image datasets achieve high accuracy [4]. The main benefits to transfer learning are seen when there is a _lack of data_ or when there is a _lack of computing power_.

In these two situations the developer may look into transfer learning to find predefined weights which can be fine-tuned to be used on a new image dataset. This will reduce overall training time and also reduce the need for a large dataset. The reason this works can be seen in [5], the early layers in a trained neural network identify lines, edges, curves. It is not until deeper in the network when objects from the training set can be recognised. Clearly trained networks can be used on other image datasets, as in all contexts the first few layers will be similar, from this point we can fine-tune the deeper layers to fit the new context.  

From a computing power perspective, it requires less time to train the final few layers and fine-tune earlier layers, from a dataset size perspective, less data is required as the earlier layers have mostly already been done for you avoiding overfitting these layers to your dataset.

# References
[1] Olga Russakovsky*, Jia Deng*, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, Alexander C. Berg and Li Fei-Fei. (* = equal contribution). _ImageNet Large Scale Visual Recognition Challenge_. IJCV. 2015.

[2] Dave Gershgorn. _The Quartz guide to artificial intelligence: What is it, why is it important, and should we be afraid?_. Quartz, 2017. Retrieved April 2020.

[3] Halevy, A., Norvig, P., Pereira, F. The Unreasonable Effectiveness of Data. _IEEE Intelligent Systems_. 2009, __24__(2), pp. 8-12.

[4] Simon Kornblith, Jonathon Shlens and Quoc V. Le. _Do Better ImageNet Models Transfer Better?_. arXiv, 2019. arXiv:1805.08974.

[5] Olah, et al.,. _Feature Visualization_. Distill, 2017.
