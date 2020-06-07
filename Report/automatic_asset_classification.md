# Automatic Asset Classification
One area of image research is called _image clustering_. This refers to finding natural groupings of unlabelled images.

Using an unsupervised algorithm for this application could lead to new classifications for assets or even help create multiple labels of assets such as the type of asset, the material of the asset and the environment the asset is located. This can help modellers deteriorate assets accurately and correctly by providing more information about assets without having to do site visits.

The other benefit to this algorithm is it can help provide initial labels if the current data does not have any.

## Pretrainined Encoder
Similar to transfer learning introduced [earlier](#transfer-learning) in the report, a pretrained encoder uses a pretrained network (such as a ResNet) and removes the head (the last few layers used for classification). This will lead to an output of a wide vector, using a ResNet this can be a length 1024, a clustering algorithm can then be applied on this output vector. [7] showed that using this technique was effective at clustering tools for use in robotics, this application builds on ideas from [8] that pretrained networks on large image sets can be fine-tuned to other tasks and help improve generalisation and at least provide better than random initialisation.

The idea is that a pretrained network will extract features throughout the network and the final large vector layer will contain representations of the extracted features which can then be clustered on. This is because when fine-tuning a network, usually the main training comes from training ahead on a pretrained network to work for the new task, the main body of the network stays almost the same and is trained at a low learning rate.

## Autoencoders
An _autoencoder_ is a neural network which learns to recreate its input usually through some bottleneck. Autoencoders have been an interest in research in deep learning since the 80s [9]. Creating a bottleneck means that autoencoders are built for dimensionality reduction, however, have also been used for anomaly detection [10] and neural network pretraining [11].

Because the encoder part of an autoencoder learns to create a smaller representation of the original input, a clustering algorithm can be applied to the bottleneck. The thought process behind this is similar to that of the [Pretrained Encoder](#pretrained-encoder), when trained to recreate the input, the encoder extracts import features from the images.

## Clustering Algorithms
_Agglomerative Hierarchical Clustering_ is used on the encoded vectors. This works by initially sorting each vector image into its cluster and then iteratively merging clusters based on minimising a distance metric. Hierarchical clustering can be plotted on a dendrogram allowing the user to decide on the optimal number of clusters.

# References
[7] Joris Guérin, Olivier Gibaru, Stéphane Thiery and Eric Nyiri.[E-Print]. CNN features are also great at unsupervised classification. _arXiv_.

[8] J. Yosinski, J. Clune, Y. Bengio, and H. Lipson, How transferable are features in deep neural networks? _Advances in neural information processing systems_. 2014, pp. 3320–3328.

[9] Ian Goodfellow, Yoshua Bengio and Aaron Courville. _Deep Learning_. MIT Press, 2016. http://www.deeplearningbook.org

[10] Chong Zhou and Randy C. Paffenroth. _Anomaly Detection with Robust Deep Autoencoders_. 2017. Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pp. 665-674. DOI: 10.1145/3097983.3098052.

[11] Dumitru Erhan, Yoshua Bengio, Aaron Courville, Pierre-Antoine Manzagol, Pascal Vincent and Samy Bengio. _Why Does Unsupervised Pre-training Help Deep Learning?_. Journal of Machine Learning Research 11 (2010), pp. 625-660.
