# Key Takeaways:
- Asset recognition proves high accuracy through the use of transfer learning.
- An experimental idea to allow algorithms to cluster assets to find underlying attributes to the assets.

# Contents
- [Introduction](#introduction)
- [Data](#data1)
  - [Data Collection](#data-collection)
  - [Data Cleaning](#data-cleaning)
- [Semi-Automatic Asset Classification](#semi-automatic-asset-classification)
  - [Transfer Learning](#transfer-learning)
- [Automatic Asset Classification](#automatic-asset-classification)
  - [Pretrained Encoders](#pretrained-encoders)
  - [Autoencoders](#autoencoders)
  - [Clustering Algorithms](#clustering-algorithms)
- [Experiment](#experiment)
  - [Data](#data2)
  - [Models](#models)
  - [Results and Discussion](#results-and-discussion)
- [Conclusion](#conclusion)
- [References](#references)

# Introduction
This project aims to automate the task of labelling images of assets, this is done by introducing two methods, _Semi-Automatic Asset Classification_ and _Automatic Asset Classification_. Semi-Automatic Asset Classification applies modern best standards of image classification on labelled images on water assets, this provides a way to automate labelling images providing some whereas Automatic Asset Classification is an experimental idea in which an algorithm clusters images of assets. The result may not lead to assets being grouped as currently labelled but could lead to more natural groupings of assets instead of just by how humans have labelled them.

# Data
The section is going to go through the collection, cleaning and preparation of data ready for the modelling. For this experiment, images of flood and water assets were collected for analysis.

The final dataset consisted of the following categories:
- Embankments
- Flood Gates
- Flood Walls
- Outfalls
- Reservoirs
- Weirs

## Data Collection
Images for the model were collected from google images. This was done with the following technique:

1. Go to google images
2. Search your key word. For example "flood gate"
3. Go to the very bottom of the page (press load more until you cannot anymore)
4. Open the console on your web browser and enter: `urls = Array.from(document.querySelectorAll('.rg_di.rg_meta')).map(el=>JSON.parse(el.textContent).ou); window.open('data:text/csv;charset=utf-8,' + escape(urls.join('\n')));`
5. Save the CSV

This gives CSV files containing image URL links. This is an effective method for smaller datasets, however, when creating a larger dataset a web scraping algorithm for image links would be more effective.

The CSV can then be used to download images using the python package `requests`.

```python
import requests
result = requests.get(image_url, stream = True)
```

## Data Cleaning
Cleaning the images ready for use in the final model consisted of three parts: removing duplicate images, removing incorrectly labelled images (from the web scrape) and finally cropping images ready for use.

### Removing Duplicate Images
To remove duplicate images multiple algorithms were used. The first check was to make sure images were not broken, this could be from corrupted file downloads. This was done with a `try` and `except` block in python by attempting to open all the images. With the broken images removed, the images could then be analysed for duplicates.

#### Hash
The first check was for images that are the same. To do this a _hash function_ was used. A hash function is a function which is a many to one mapping in which inputs of arbitrary length all output to a fixed length [1]. Even though this is a one to many mapping, a "good" hash function it is computationally infeasible to find two distinct inputs mapping to the same output.

Images ran through the hash function to create the hash representation any images with the same hash the duplicate images were dropped.

#### Dhash
Once done another type of hash algorithm called _dhash_ (difference hash) was performed. This removes similar images instead of the same images. To do this images are resized to be the same size and then re-coloured to black and white. The gradient of the black and white image is then calculated along each axis, resulting in a matrix of 1s and 0s representing changes in intensities in the image. This is then fed into a hash function again and the same method as before is used to remove images.

#### Hamming Distance
Finally, _hamming distance_ is used to find two similar images with a threshold. Hamming distance counts the number of differences between two arrays of the same sized. From there you can calculate a percentage in which the images are different in and set a threshold for example 10%.

Example of hamming distance:
```
x = [1,5,6,9]
y = [2,5,6,2]
hamming_distance(x,y) = 1+0+0+1 = 2
percentage_difference = 2/4 = 0.5
```

### Removing Incorrect Images
Removing incorrect images was done in a semi-automated fashion. For each category of image (such as Flood Wall), the first 50 or so images were labelled as a "yes" or "no" representing whether the current label for the images was correct. This was done through fine-tuning a pretrained Resnet34 model from [PyTorch](https://pytorch.org/hub/pytorch_vision_resnet/).

Once trained on the first 50 images, the model can then be used to predict whether the remaining images are correctly labelled. If the model predicted "yes" the image was kept and if "no" the image was removed.

### Image Cleaning
After all the above processing, the dataset left is _almost_ final. The last step to prepare the data ready for use is to crop the images so that the object is centred is and so that all images are square. This is all done by hand as to avoid issues with automated algorithms centring on the wrong object arising. While doing this, images can be checked if they were correctly labelled too.

This is done to make sure that all algorithms focus on the correct object in the image (in case an image contains multiple objects). Images are squared as it means that pretrained architectures can be used.

# Semi-Automatic Asset Classification
Image classification and object identification with neural networks has been proven very effect. In the 2017 ImageNet Competition <sup>[2](#imageNetFootNote)</sup> 29 of the 38 teams competing achieved an accuracy of greater than 95% [2][3]

Semi-Automatic Asset Classification aims to apply these top standards to asset classification. This is done by training a neural network on a set of _labelled images_. The issue with this type of modelling is a large dataset of already labelled images is required, this can be costly and timely to create. To reach the top accuracy it is argued that more data is more important than better modelling [4], meaning that to keep improving models more data is required. Here, _transfer learning_ can be used to reduce this need or such large datasets.

## Transfer Learning
_Transfer learning_ is the use of applying one neural network to a new problem, the hope is that the learned architecture can be applied to a new problem. This technique has been proven efficient, especially in image classification where pretrained ImageNet models are transferred and fine-tuned to new image datasets achieve high accuracy [5]. The main benefits to transfer learning are seen when there is a _lack of data_ or when there is a _lack of computing power_.

In these two situations, the developer may look into transfer learning to find predefined weights which can be fine-tuned to be used on a new image dataset. This will reduce overall training time and also reduce the need for a large dataset. The reason this works can be seen in [6], the early layers in a trained neural network identify lines, edges, curves. It is not until deeper in the network when objects from the training set can be recognised. Trained networks can be used on other image datasets, as in all contexts the first few layers will be similar, from this point we can fine-tune the deeper layers to fit the new context.  

From a computing power perspective, it requires less time to train the final few layers and fine-tune earlier layers, from a dataset size perspective, less data is required as the earlier layers have mostly already been done for you avoiding overfitting these layers to your dataset.

# Automatic Asset Classification
One area of image research is called _image clustering_. This refers to finding natural groupings of unlabelled images.

Using an unsupervised algorithm for this application could lead to new classifications for assets or even help create multiple labels of assets such as the type of asset, the material of the asset and the environment the asset is located. This can help modellers deteriorate assets accurately and correctly by providing more information about assets without having to do site visits.

The other benefit to this algorithm is it can help provide initial labels if the current data does not have any.

## Pretrained Encoder
Similar to transfer learning introduced [earlier](#transfer-learning) in the report, a pretrained encoder uses a pretrained network (such as a ResNet) and removes the head (the last few layers used for classification). This will lead to an output of a wide vector, using a ResNet this can be a length 1024, a clustering algorithm can then be applied on this output vector. [7] showed that using this technique was effective at clustering tools for use in robotics, this application builds on ideas from [8] that pretrained networks on large image sets can be fine-tuned to other tasks and help improve generalisation and at least provide better than random initialisation.

The idea is that a pretrained network will extract features throughout the network and the final large vector layer will contain representations of the extracted features which can then be clustered on. This is because when fine-tuning a network, usually the main training comes from training ahead on a pretrained network to work for the new task, the main body of the network stays almost the same and is trained at a low learning rate.

## Autoencoders
An _autoencoder_ is a neural network which learns to recreate its input usually through some bottleneck. Autoencoders have been an interest in research in deep learning since the 80s [9]. Creating a bottleneck means that autoencoders are built for dimensionality reduction, however, have also been used for anomaly detection [10] and neural network pretraining [11].

Because the encoder part of an autoencoder learns to create a smaller representation of the original input, a clustering algorithm can be applied to the bottleneck. The thought process behind this is similar to that of the [Pretrained Encoder](#pretrained-encoder), when trained to recreate the input, the encoder extracts import features from the images.

## Clustering Algorithms
_Agglomerative Hierarchical Clustering_ is used on the encoded vectors. This works by initially sorting each vector image into its cluster and then iteratively merging clusters based on minimising a distance metric. Hierarchical clustering can be plotted on a dendrogram allowing the user to decide on the optimal number of clusters.

# Experiment
All experiments were run on [Google Colab](https://colab.research.google.com/notebooks/welcome.ipynb) using an assigned GPU at that time, this could mean some models may have used a better GPU than other models, however, this will only have an effect on the time taken and not the outputs.

Due to using a free online GPU provided by Google, the amount of training is minimal (while still providing good results) as to not take advantage or exploit this product.

## Data
The final dataset consists of classes:
- Embankment
- Flood Gate
- Flood Wall
- Outfall
- Reservoir
- Weir

With a training dataset length of 444 and a validation dataset of length 110.

## Models
For the semi-automatic asset classification, a Resnet34 pretrained model is fine-tuned to fit to the asset image dataset.
[image of models](here)

A basic overview of the architecture of the three models used for clustering can be seen in the diagram. The code for these are available on the [Github repository](h). _Pretrained Encoder_ uses the pretrained Resnet34 body as an encoder, _Basic Autoencoder_ uses a simple architecture and _Resnet Autoencoder_ uses the pretrained Resnet34 as an encoder and a decoder is trained on top of this.

Before clustering is applied, _principal component analysis_ is applied to reduce vector of length 1024 to a vector of length 50. This further reduces the dimension of the vector in a __linear__ way.

## Results and Discussion
The models are trained and the results from the __validation__ set (unseen data in training) are discussed here.

### Semi-Automatic Asset Classification
A classification accuracy of 0.9000 (90%) is achieved after training for 25 epochs. This is a very high accuracy and shows potential for models to be very accurate for more clean (official) datasets.

The confusion matrix shows that most incorrect classifications are related to the flood wall class. This is mainly due to assets similar to flood walls appearing close to other assets such as on large flood gates or near embankments. It could be argued that with a cleaner dataset that these sort of errors could be taken out. When web-scraping for images it is common to find multiple assets in some of the images which can lead to confusion in classification.

Looking at the diagram of the 9 highest losses from the model, this further supports this showing that a lot of misclassifications could be due to poor imagery. The top-left image shows a long outfall pipe in the ocean, however, due to the large pool of water the model has learned to associate pools of water with reservoirs. A better image would focus more on the individual asset, perhaps more up close.

### Automatic Asset Classification
The results for automatic asset classification is difficult to put into quantitative terms and results are more subjective. The quantitative result which will be mentioned for each model is the optimal number of clusters, this shows how well the models were at dividing the data into clusters. Beliefs of how the models have clustered data will be discussed.

Plots of the clusters are shown with TSNE dimensionality reduction. These are done with "perplexity" 5, 30 and 50 to get a good overview of how separated the data is.


#### Pretrained Encoder
The pretrained Resnet encoder was able to split the data into 3 clusters. As can be seen, the clusters appear to be very intertwined with each other suggesting not a good split.

The first cluster is primarily outfall images. 10/17 images are of an outfall in this cluster, suggesting the model learned a strong relationship between those images. All of the remaining images are featured in a green or muddy area suggesting the model found a good environmental relationship. However, there is not much to associate between them.

The second cluster showed primarily large bodies of water and was predominately images of reservoirs, however there we also many images of greenery from embankments. The difference between the green areas from cluster 1 and 2 is that there is no livestock in cluster 2. This shows that the model could help in categorising use of assets. There were no other assets in cluster 2 except embankments and reservoirs.

The third cluster showed assets which were your more aggressive defences, ones in water like weirs and flood gates and large structures like flood walls made out of concrete or metal. This cluster could be assets which are more likely to take damage.

#### Basic Autoencoder
In terms of the optimal number of clusters, the basic autoencoder managed the least amount, only splitting the data into two clusters. However as can be seen from the TSNE plots, this is more clear split than what was achieved with the pretrained encoder.

The two clusters in the basic autoencoder are hard to find a relationship between without forcing one. Each cluster contains an array of all asset types and environment types. This suggests this model was not able to find divisive features in the asset images.


#### Resnet Autoencoder
The full ResNet autoencoder was the best in terms of the optimal number of clusters, achieving 5 clusters.

Cluster 1 contained the majority of images and it is hard to decipher any underlying attributes between the assets, this could be thought of a cluster created when the images do not fall into any of the other clusters.

Cluster 2 contained images which has flowing water, suggesting the model could have grouped these based on damages which could be caused from flowing water. This included primarily weirs, reservoirs and outfalls. This cluster contained 24 images.

Cluster 3 was a small cluster and only contained 8 images, 6 of these we embankments with the remaining two being a reservoir and a flood gate.

Cluster 4 also only contained 8 images. 3 of these were weirs, 3 were flood walls, 1 was a flood gate and 1 was an embankment. Many of these images also contain the light coloured stone used in some bricks and walls suggesting the model was perhaps relating the material to these together.

Cluster 5 contained 13 assets, and was primarily metal flood gates or floodwalls. They also appear to be assets with long lives and this could be a benefit to group them. Other assets included embankments and reservoirs.

# Conclusion
In conclusion, this report has developed a proof of concept for the use of autoencoders in classifying flood assets. This could be extended to include assets from all domains. Semi-Automatic Asset Classification has shown that with labelled data neural networks can be trained to classify assets to a high accuracy of 90%.

With unsupervised training, this report has shown neural networks can group assets by some underlying features such as material, environment and risk. 3 models were compared for this section and a Resnet autoencoder proved to be most effective, being able to split up the asset images into more clusters leading to more underlying features. These clusters also had more clear underlying features linking the assets within them together.

Overall, a combination of both Semi-Automatic and Automatic Asset Classification could be most effective. With Semi-Automatic Asset Classification able to classify assets to their high-level labels which could be combined with Semi-Automatic Asset Classification to add more information to these assets from underlying features found between them.

One of the main drawbacks of this experiment was the size and quality of the dataset. It is hard to find high-quality images of flood assets online and perhaps if this experiment was repeated in-house at a flood defence company better results could be achieved with more clear, higher quality images and a larger set of images.

There are two clear ways to extend this experiment. One way would be to include a dataset with both high level and low-level labels leading to a supervised way to train a network to predict both high-level labels such as asset type and low-level features such as material. The other way would be to use image segmentation to develop a model which could label _all_ assets in one image instead of being limited to one asset per image.

# References
[1] Handbook of applied cryptography.  Menezes, A. J. (Alfred J.), (1997). Boca Raton: CRC Pre

[2] Olga Russakovsky*, Jia Deng*, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, Alexander C. Berg and Li Fei-Fei. (* = equal contribution). _ImageNet Large Scale Visual Recognition Challenge_. IJCV. 2015.

[3] Dave Gershgorn. _The Quartz guide to artificial intelligence: What is it, why is it important, and should we be afraid?_. Quartz, 2017. Retrieved April 2020.

[4] Halevy, A., Norvig, P., Pereira, F. The Unreasonable Effectiveness of Data. _IEEE Intelligent Systems_. 2009, __24__(2), pp. 8-12.

[5] Simon Kornblith, Jonathon Shlens and Quoc V. Le. _Do Better ImageNet Models Transfer Better?_. arXiv, 2019. arXiv:1805.08974.

[6] Olah, et al.,. _Feature Visualization_. Distill, 2017.

[7] Joris Guérin, Olivier Gibaru, Stéphane Thiery and Eric Nyiri.[E-Print]. CNN features are also great at unsupervised classification. _arXiv_.

[8] J. Yosinski, J. Clune, Y. Bengio, and H. Lipson, How transferable are features in deep neural networks? _Advances in neural information processing systems_. 2014, pp. 3320–3328.

[9] Ian Goodfellow, Yoshua Bengio and Aaron Courville. _Deep Learning_. MIT Press, 2016. http://www.deeplearningbook.org

[10] Chong Zhou and Randy C. Paffenroth. _Anomaly Detection with Robust Deep Autoencoders_. 2017. Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pp. 665-674. DOI: 10.1145/3097983.3098052.

[11] Dumitru Erhan, Yoshua Bengio, Aaron Courville, Pierre-Antoine Manzagol, Pascal Vincent and Samy Bengio. _Why Does Unsupervised Pre-training Help Deep Learning?_. Journal of Machine Learning Research 11 (2010), pp. 625-660.
