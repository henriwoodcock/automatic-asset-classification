from fastai.vision import *
image_path = os.getcwd() + "/data/processing/embankment/"

data = ImageDataBunch.from_folder(image_path, valid_pct = 0.2, size=224,ds_tfms=get_transforms()).normalize(imagenet_stats)

learn = cnn_learner(data, models.resnet34, metrics=error_rate)
learn.fit_one_cycle(4)

interp = ClassificationInterpretation.from_learner(learn)
losses,idxs = interp.top_losses()
len(data.valid_ds)==len(losses)==len(idxs)

interp.plot_top_losses(9, figsize=(15,11))
interp.plot_confusion_matrix(figsize=(12,12), dpi=60)
