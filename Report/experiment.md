# Experiment
All experiments were run on [Google Colab](https://colab.research.google.com/notebooks/welcome.ipynb) using an assigned GPU at that time, this could mean some models may have used a better GPU than other models, however this will only have an effect on time taken and not the outputs.

Due to using a free online GPU provided by Google, amount of training is minimal (while still providing good results) as to not take advantage or exploit this product.

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

A basic overview of the architecture of the three models used for clustering can be seen in the diagram. The code for these are available on the [Github repository](h). _Pretrained Encoder_ uses the pretrained Resnet34 body as an encoder, _Basic Autoencoder_ uses a simple architecture and _Resnet Autoencoder_ uses the pretrained Resnet34 as an encoder and a decoder is trained ontop of this.

Before clustering is applied, _principal component analysis_ is applied to reduce vector of length 1024 to a vector of length 50. This further reduces the dimension of the vector in a __linear__ way.

## Results and Discussion
The models are trained and the results from the __validation__ set (unseen data in training) are discussed here.

### Semi-Automatic Asset Classification
A classification accuracy of 0.9000 (90%) is achieved after training for 25 epochs. This is a very high accuracy and shows potential for models to be very accurate for more clean (official) datasets.

The confusion matrix shows that most incorrect classifications are related to the flood wall class. This is mainly due to assets similar to flood walls appearing close to other assets such as on large flood gates or near embankments. It could be argued that with a cleaner dataset that these sort of errors could be taken out. When webscraping for images it is common to find multiple assets in some of the images which can lead to confusion in classification.

Looking at the diagram of the 9 highest losses from the model, this further supports this showing that a lot of misclassifications could be due to poor imagery. The top-left image shows a long outfall pipe in the ocean, however due to the large pool of water the model has learned to associate pools of water with reservoirs. A better image would focus more on the individual asset, perhaps more upclose.

### Automatic Asset Classification
The results for automatic asset classification is difficult to put into quantitative terms and results are more subjective. The quantitative result which will be mentioned for each model is the optimal number of clusters, this shows how well the models were at dividing the data into clusters. Beliefs of how the models have clustered data will be discussed.

Plots of the clusters are shown with TSNE dimensionality reduction. These are done with "perplexity" 5, 30 and 50 to get a good overview of how separated the data is.


#### Pretrained Encoder
The pretrained Resnet encoder was able to split the data into 3 clusters. As can be seen the clusters appear to be very intertwined with each other suggesting not a good split.

The first cluster is primarily outfall images. 10/17 images are of an outfall in this cluster, suggesting the model learned a strong relationship between those images. All of the remaining images are featured in a green or muddy area suggesting the model found a good environmental relationship. However there is not much to associate between them.

The second cluster showed primarily large bodies of water and was predominately images of reservoirs, however there we also many images of greenery from embankments. The difference between the green areas from cluster 1 and 2 is that there are no livestock in cluster 2. This shows that the model could help in categorising use of assets. There were no other assets in cluster 2 except embankments and reservoirs.

The third cluster showed assets which were your more agressive defences, ones in water like weirs and flood gates and large structures like flood walls made out of concrete or metal. This cluster could be assets which are more likely to take damage.

#### Basic Autoencoder
In terms of optimal number of clusters, the basic autoencoder managed the least amount, only splitting the data into two clusters. However as can be seen from the TSNE plots, this is more clear split than what was achieved with the pretrained encoder.

The two clusters in the basic autoencoder are hard to find a relationship between without forcing one. Each cluster contains an array of all asset types and environment types. This suggests this model was not able to find divisive features in the asset images.


#### Resnet Autoencoder
The full resnet autoencoder was the best in terms of optimal number of clusters, achieving 5 clusters.

Cluster 1 contained the majority of images and it is hard to decipher any underlying attributes between the assets, this could be thought of a cluster created when the images do not fall into any of the other clusters.

Cluster 2 contained images which has flowing water, suggesting the model could have grouped these based on damages which could be caused from flowing water. This included primarily weirs, reservoirs and outfalls. This cluster contained 24 images.

Cluster 3 was a small cluster and only contained 8 images, 6 of these we embankments with the remaining two being a reservoir and a flood gate.

Cluster 4 also only contained 8 images. 3 of these were weirs, 3 were flood walls, 1 was a flood gate and 1 was an embankment. Many of these images also contain the light coloured stone used in some bricks and walls suggesting the model was perhaps relating the material to these together.

Cluster 5 contained 13 assets, and was primarily metal flood gates or flood walls. They also appear to be assets with long lives and this could be a benefit to group them together. Other assets included embankments and reservoirs.

# Conclusion
In conclusion this report has developed a proof of concept for the use of autoencoders in classifying flood assets. This could be extended to include assets from all domains. Semi-Automatic Asset Classification has shown that with labelled data neural networks can be trained to classify assets to a high accuracy of 90%.

With unsupervised training, this report has shown neural networks are able to group assets by some underlying features such as material, environment and risk. 3 models were compared for this section and a Resnet autoencoder proved to be most effective, being able to split up the asset images into more clusters leading to more underlying features. These clusters also had more clear underlying features linking the assets within them together.

Overall, a combination of both Semi-Automatic and Automatic Asset Classification could be most effective. With Semi-Automatic Asset Classification able to classify assets to their high-level labels which could be combined with Semi-Automatic Asset Classification to add more information to these assets from underlying features found between them.

One of the main drawbacks of this experiment was the size and quality of the dataset. It is hard to find high quality images of flood assets online and perhaps if this experiment was repeated in house at a flood defense company better results could be achieved with more clear, higher quality images and a larger set of images.

There are two clear ways to extend this experiment. One way would be to include a dataset with both high level and low level labels leading to a supervised way to train a network to predict both high level labels such as asset type and low level features such as material. The other way would be to use image segmentation to develop a model which could label _all_ assets in one image instead of being limited to one asset per image.
