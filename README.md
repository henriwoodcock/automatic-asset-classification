# automatic asset classification

## IDEAS
- flood assets
- hydraulic modelling
- pipe material classification ?
- railway assets

### Flood Assets:
- Wall
- Flood gate
- Open channel
- Embankment – trapezoidal
- Reservoir
- Outfall
- Weirs – small dams
- Mesh
- Control gate
- Channel crossing bridge

### Pipe Material Classification:
- Cast Iron
- Ductile Iron
- HPPE
- Steel
- Spun Iron


## Datasets

- autstialian trunk main dataset:
https://catalogue.data.wa.gov.au/dataset/activity/water-pipe-wcorp-002
- water treatment plant dataset:
https://archive.ics.uci.edu/ml/datasets/Water+Treatment+Plant
- Distribution input and supply pipe leakage: 1992/93 to 2010/11:
https://data.gov.uk/dataset/6f916c63-b142-4929-be09-0861b3c0ce6a/distribution-input-and-supply-pipe-leakage-1992-93-to-2010-11/datafile/eccfa62e-a7ad-4aaf-8ed9-089feee36214/preview

- uk government datasets:
https://data.gov.uk/search?q=

https://www.ibm.com/developerworks/community/wikis/home?lang=en#!/wiki/IBM%20Maximo%20Asset%20Management/page/Data%20sets



## Webscraping for Images
1. Go to google images
2. Search your key word. For example "flood gate"
3. Go to the very bottom of the page (press load more until you cannot anymore)
4. Open the console on your web browser and enter:
urls = Array.from(document.querySelectorAll('.rg_di .rg_meta')).map(el=>JSON.parse(el.textContent).ou);
window.open('data:text/csv;charset=utf-8,' + escape(urls.join('\n')));
5. Save the csv

This provides you with a database of links to images. This is easier than scraping for the links then downloading the images from the links. However feel free to scrape for the links if not using google images for your data.

From trying to classify them into correct or incorrect, the hardest one was the flood gate. A lot of the images online are from a brand called flood gate which makes it hard to find proper flood gates. This model managed to predict wrong 50% of the time as the model never predicted no. This could result in a lower quality of dataset for the final model. Rerunning this managed an accuracy of 75% however it is unsure how accurate this will be on the final dataset as the images are hard to classify even for human.

flood wall 10 could be a flood wall however when comparing to embankment it would be an embankment, so this one cannot be used in the final dataset. _However could be used in the later autoencoding section._

### Data Clean Up
A resnet34 model was then used on a basic classificiation model to further clean up the data. This involved create a resnet34 model on the "final" dataset and then plotting the top losses. This should help find images which are not clean / not clear what they are of. A few examples can be seen below. For example a weir can form over a floodwall or a image could be of a reservoir with a outfall in it. Removing these can help create a more accurate model, as if an image has multiple assets in it then this will become confusing for the model.

# Goals
Asset classification without labelling. This will be done with an auto-encoder. This could lead to a future development where AI can help put assets into categories in which humans may struggle with defining said categories. This is a basic example with some very differing examples, but the hope is that it could be used in the future on more confusing assets. For example some assets may be a flood wall made out of sheets however it has been upgraded but the sheet pile is still there. An AI could help make a conclusion as to how to classify this.

A lot of images in asset management are also labelled to just include a location of where the image was taken but not include the type of asset this is, this would then mean that a human would have to go through these images to classify them. Creating a basic classifier model could help speed this up.

test:
*hfhsh and  then __this__*

checkout fast.ai image classification on my gists!!!


https://github.com/moondra2017/Computer-Vision/blob/master/DHASH%20AND%20HAMMING%20DISTANCE_Lesson.ipynb
this is the help for the duplicate images


Have a look at Bayesian deep learning it could tell hardest assets to classify.......

imagecleaner to help clean up top losses.


## Research
transfer learning:
https://medium.com/pytorch/active-transfer-learning-with-pytorch-71ed889f08c1
self-supervised learning and computer vision:
https://www.fast.ai/2020/01/13/self_supervised/ - maybe attempt a few of these see which is best for pretraining.

https://www.biorxiv.org/content/10.1101/740548v4 - pretraining
https://chemrxiv.org/articles/Inductive_Transfer_Learning_for_Molecular_Activity_Prediction_Next-Gen_QSAR_Models_with_MolPMoFiT/9978743/1
https://github.com/kheyer/Genomic-ULMFiT
https://www.youtube.com/playlist?list=PLoRl3Ht4JOcdU872GhiYWf6jwrk_SNhz9



IDEAS for post (temporary on here):
- explainable AI: first look at weightings for singular layer AI:
  https://stats.stackexchange.com/questions/261008/deep-learning-how-do-i-know-which-variables-are-important
  https://beckmw.wordpress.com/2013/08/12/variable-importance-in-neural-networks/
  https://towardsdatascience.com/feature-importance-with-neural-network-346eb6205743
