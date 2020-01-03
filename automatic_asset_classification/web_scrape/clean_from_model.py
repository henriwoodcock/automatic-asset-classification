from fastai.vision import *
#from fastai.widgets.image_cleaner import *
import matplotlib.pyplot as plt
import os
import pandas as pd

image_path = os.getcwd() + "/data/final_dataset/final"
#create fastai data bunch for pre processing
data = ImageDataBunch.from_folder(image_path, valid_pct = 0.2, size=224,ds_tfms=get_transforms(), test = "test").normalize(imagenet_stats)
#create learner
#resnet 50?
learn = cnn_learner(data, models.resnet34, metrics=error_rate)
#fit on just end layers (other layers are froze)
learn.fit_one_cycle(4)

interp = ClassificationInterpretation.from_learner(learn)
losses,idxs = interp.top_losses()
#len(data.valid_ds)==len(losses)==len(idxs)
#validation analysis
interp.plot_top_losses(9, figsize=(15,11))
interp.plot_confusion_matrix(figsize=(12,12), dpi=60)
plt.show()
