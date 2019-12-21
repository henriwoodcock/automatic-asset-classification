from fastai.vision import *
import matplotlib.pyplot as plt

types = ["embankment", "flood_gate", "flood_wall", "outfall", "reservoir", "weir"]

learner_dict = dict()

for type_ in types:
    #get location
    image_path = os.getcwd() + "/data/processing/" + types_ + "/"
    #create fastai data bunch for pre processing
    data = ImageDataBunch.from_folder(image_path, valid_pct = 0.2, size=224,ds_tfms=get_transforms()).normalize(imagenet_stats)
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
    learner_dict[type_] = learner
#print plots of validagtion analysis
plt.show()
