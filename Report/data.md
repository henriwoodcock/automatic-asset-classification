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
5. Save the csv

This gives csv files containing image url links. This is an effective method for smaller datasets, however when creating a larger dataset a webscraping algorithm for image links would be more effective.

The csv can then be used to download images using the python package `requests`.

```python
import requests
result = requests.get(image_url, stream = True)
```

## Data Cleaning
Cleaning the images ready for us in the final model consisted of three parts: removing duplicate images, removing incorrectly labelled images (from the webscape) and finally cropping images ready for use.

### Removing Duplicate Images
To remove duplicate images multiple algorithms were used. The first check was to make sure images were not broken, this could be from corrupted file downloads. This was done with a `try` and `except` block in python by attempting to open all the images. With the broken images removed, the images could then be analysed for duplicates.

#### Hash
The first check was for images that are exactly the same. To do this a _hash function_ was used. A hash function is a function which is a many to one mapping in which inputs of arbitrary length all output to a fixed length [1]. Even though this is a one to many mapping, a "good" hash function it is computationally infeasible to find two distinct inputs mapping to the same output.

Images were ran through the has function to create the hash representation any images with the same hash the duplicate images were dropped.

#### Dhash
Once done another type of hash algorithm called _dhash_ (difference hash) was performed. This removes similar images instead of exactly the same images. To do this images are resized to be the same size and then re-coloured to black and white. The gradient of the black and white image is then calculated along each axis, resulting in a a matrix of 1s and 0s representing changes in intensities in the image. This is then fed into a hash function again and the same method as before is used to remove images.

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
After all the above processing, the dataset left is _almost_ final. The laat step to prepare the data ready for use is to crop the images so that the object is centred is and so that all images are square. This is all done by hand as to avoid issues with automated algorithms centering on the wrong object arising. While doing this, images can be checked if they were correctly labelled too.

This is done to make sure that all algorithms focus on the correct object in the image (incase an image contains multiple objects). Images are squared as it means that pretrained architectures can be used.

## Referneces
[1] Handbook of applied cryptography.  Menezes, A. J. (Alfred J.), (1997). Boca Raton : CRC Pre
