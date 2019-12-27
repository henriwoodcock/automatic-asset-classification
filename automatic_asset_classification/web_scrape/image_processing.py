from fastai.vision import *
import matplotlib.pyplot as plt
import os

types = ["embankment", "flood_gate", "flood_wall", "outfall", "reservoir", "weir"]

learner_dict = dict()

for type_ in types:
    #get location
    image_path = os.getcwd() + "/data/processing/" + type_ + "/"
    #create fastai data bunch for pre processing
    data = ImageDataBunch.from_folder(image_path, valid_pct = 0.2, size=224,ds_tfms=get_transforms(), test = "test").normalize(imagenet_stats)
    #create learner
    #resnet 50?
    learn = cnn_learner(data, models.resnet34, metrics=error_rate)
    #fit on just end layer (other layers are froze)
    learn.fit_one_cycle(4)
    #interprate class
    interp = ClassificationInterpretation.from_learner(learn)
    losses,idxs = interp.top_losses()
    len(data.valid_ds)==len(losses)==len(idxs)
    #validation analysis
    interp.plot_top_losses(9, figsize=(15,11))
    interp.plot_confusion_matrix(figsize=(12,12), dpi=60)
    #add learner to dictionary so can be used for prediction later
    learner_dict[type_] = learn
#print plots of validagtion analysis
plt.show()

for type_ in types:
    #to delete file rm from folder
    path_ = os.getcwd() + "/" + type_
    #if file not right then put in no
    if learner.predict(type_) == 1:
        type_ = (os.cwd() = os.cwd() + "/" + "no/"
    #if file right then add to yes
    if leaner.predict(type_) == 0:
        type_ = (os.cwd() = os.cwd() + "/" + "yes/"

#doing them on by one:
data = ImageList.from_folder(loc).normalize(imagenet_stats)

#create a function:
def create_yes_no_model(type_):
    image_path = os.getcwd() + "/data/processing/" + type_ + "/"
    #create fastai data bunch for pre processing
    data = ImageDataBunch.from_folder(image_path, valid_pct = 0.2, size=224,ds_tfms=get_transforms(), test = "test").normalize(imagenet_stats)
    #create learner
    #resnet 50?
    learn = cnn_learner(data, models.resnet50, metrics=error_rate)
    #fit on just end layer (other layers are froze)
    learn.fit_one_cycle(4)
    #interprate class
    interp = ClassificationInterpretation.from_learner(learn)
    losses,idxs = interp.top_losses()
    len(data.valid_ds)==len(losses)==len(idxs)
    #validation analysis
    interp.plot_top_losses(9, figsize=(15,11))
    interp.plot_confusion_matrix(figsize=(12,12), dpi=60)
    plt.show()
    return learn



type_ = "embankment"
#embank_leanr = create_yes_no_model(type_)

image_path = os.getcwd() + "/data/processing/" + type_ + "/"
#create fastai data bunch for pre processing
data = ImageDataBunch.from_folder(image_path, valid_pct = 0.2, size=224,ds_tfms=get_transforms(), test = "test").normalize(imagenet_stats)
#create learner
#resnet 50?
learn = cnn_learner(data, models.resnet34, metrics=error_rate)
#fit on just end layer (other layers are froze)
learn.fit_one_cycle(4)
#interprate class
interp = ClassificationInterpretation.from_learner(learn)
losses,idxs = interp.top_losses()
len(data.valid_ds)==len(losses)==len(idxs)
#validation analysis
interp.plot_top_losses(9, figsize=(15,11))
interp.plot_confusion_matrix(figsize=(12,12), dpi=60)
plt.show()


preds,y = embank_leanr.get_preds(ds_type=DatasetType.Test)

data.test_dl.dl.dataset.x.items

img = embank_leanr.data.train_ds[0][0]
img.show()
#predictions
path2 = os.getcwd() + "/data/raw/" + type_ + "/"
data.add_test_folder(path2)
data2 = ImageDataBunch.from_folder(path2, valid_pct = 0.2, size=224,ds_tfms=get_transforms()).normalize(imagenet_stats)

learn.data.test_dl.dl.dataset.x.items
predictions = []
for i in range(len(data.test_ds.x)):
    img = embank_leanr.data.train_ds[i][0]
    predictions.append(learn.predict(img)[0])
