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



checkout fast.ai image classification on my gists!!!


https://github.com/moondra2017/Computer-Vision/blob/master/DHASH%20AND%20HAMMING%20DISTANCE_Lesson.ipynb
this is the help for the duplicate images


Have a look at Bayesian deep learning it could tell hardest assets to classify.......

imagecleaner to help clean up top losses.
