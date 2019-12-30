from fastai.vision import *
import matplotlib.pyplot as plt
import os
import pandas as pd

types = ["embankment", "flood_gate", "flood_wall", "outfall", "reservoir", "weir"]

test_path = os.getcwd() + "/data/raw/"
out_path = os.getcwd() + "/data/processing/"
learner_dict = dict()

#create a function:
def create_yes_no_model(type_):
    image_path = os.getcwd() + "/data/processing/" + type_ + "/"
    #create fastai data bunch for pre processing
    data = ImageDataBunch.from_folder(image_path, valid_pct = 0.2, size=224,ds_tfms=get_transforms(), test = "test").normalize(imagenet_stats)
    #create learner
    #resnet 50?
    learn = cnn_learner(data, models.resnet34, metrics=error_rate)
    #fit on just end layers (other layers are froze)
    learn.fit_one_cycle(5)
    return learn

def plot_val(learn):
    #interprate class
    interp = ClassificationInterpretation.from_learner(learn)
    losses,idxs = interp.top_losses()
    #len(data.valid_ds)==len(losses)==len(idxs)
    #validation analysis
    interp.plot_top_losses(9, figsize=(15,11))
    interp.plot_confusion_matrix(figsize=(12,12), dpi=60)
    plt.show()

def return_classes(zeros, ones, pred):
    if pred == 0:
        return zeros
    else:
        return ones

#embankment
type_ = "embankment"
embank_learn = create_yes_no_model(type_)
#check accuracy on validation set:
plot_val(embank_learn)
img = embank_learn.data.train_ds[0][0]
img.show()
plt.show()
embank_learn.data.classes
#category "no" = 0
#just to check:
embank_learn.predict(img)
#now predict test set.
preds,y = embank_learn.get_preds(ds_type=DatasetType.Test)
pred_out = np.argmax(preds, axis = 1).numpy()
#convert to classes
pred_class = [return_classes("no", "yes", pred) for pred in pred_out]
#grab file names
labels = embank_learn.data.test_dl.dl.dataset.x.items
#output
df = pd.DataFrame({"Filename": labels, "Correct": pred_class})
df.to_csv(out_path + str(type_) + "folder_pred.csv", index = False)

#flood_gate
type_ = "flood_gate"
gate_learn = create_yes_no_model(type_)
#check accuracy on validation set:
plot_val(gate_learn)
img = gate_learn.data.train_ds[0][0]
img.show()
plt.show()
gate_learn.data.classes
#category "no" = 0
#just to check:
gate_learn.predict(img)
#now predict test set.
preds,y = gate_learn.get_preds(ds_type=DatasetType.Test)
pred_out = np.argmax(preds, axis = 1).numpy()
#convert to classes
pred_class = [return_classes("no", "yes", pred) for pred in pred_out]
#grab file names
labels = gate_learn.data.test_dl.dl.dataset.x.items
#output
df = pd.DataFrame({"Filename": labels, "Correct": pred_class})
df.to_csv(out_path + str(type_) + "folder_pred.csv", index = False)
gate_learn.save(out_path + str(type_) + "_learner")

#flood wall
type_ = "flood_wall"
wall_learn = create_yes_no_model(type_)
#check accuracy on validation set:
plot_val(wall_learn)
wall_learn.data.classes
#category "no" = 0
#just to check:
wall_learn.predict(img)
#now predict test set.
preds,y = wall_learn.get_preds(ds_type=DatasetType.Test)
pred_out = np.argmax(preds, axis = 1).numpy()
#convert to classes
pred_class = [return_classes("no", "yes", pred) for pred in pred_out]
#grab file names
labels = wwall_learn.data.test_dl.dl.dataset.x.items
#output
df = pd.DataFrame({"Filename": labels, "Correct": pred_class})
df.to_csv(out_path + str(type_) + "folder_pred.csv", index = False)
wall_learn.save(out_path + str(type_) + "_learner")

#outfall
type_ = "outfall"
outfall_learn = create_yes_no_model(type_)
#check accuracy on validation set:
plot_val(outfall_learn)
outfall_learn.data.classes
#category "no" = 0
#just to check:
outfall_learn.predict(img)
#now predict test set.
preds,y = outfall_learn.get_preds(ds_type=DatasetType.Test)
pred_out = np.argmax(preds, axis = 1).numpy()
#convert to classes
pred_class = [return_classes("no", "yes", pred) for pred in pred_out]
#grab file names
labels = outfall_learn.data.test_dl.dl.dataset.x.items
#output
df = pd.DataFrame({"Filename": labels, "Correct": pred_class})
df.to_csv(out_path + str(type_) + "folder_pred.csv", index = False)
outfall_learn.save(out_path + str(type_) + "_learner")

#reservoir
type_ = "reservoir"
reservoir_learn = create_yes_no_model(type_)
#check accuracy on validation set:
plot_val(reservoir_learn)
reservoir_learn.data.classes
#category "no" = 0
#just to check:
reservoir_learn.predict(img)
#now predict test set.
preds,y = reservoir_learn.get_preds(ds_type=DatasetType.Test)
pred_out = np.argmax(preds, axis = 1).numpy()
#convert to classes
pred_class = [return_classes("no", "yes", pred) for pred in pred_out]
#grab file names
labels = reservoir_learn.data.test_dl.dl.dataset.x.items
#output
df = pd.DataFrame({"Filename": labels, "Correct": pred_class})
df.to_csv(out_path + str(type_) + "folder_pred.csv", index = False)
reservoir_learn.save(out_path + str(type_) + "_learner")

#weir
type_ = "weir"
weir_learn = create_yes_no_model(type_)
#check accuracy on validation set:
plot_val(weir_learn)
weir_learn.data.classes
#category "no" = 0
#just to check:
weir_learn.predict(img)
#now predict test set.
preds,y = weir_learn.get_preds(ds_type=DatasetType.Test)
pred_out = np.argmax(preds, axis = 1).numpy()
#convert to classes
pred_class = [return_classes("no", "yes", pred) for pred in pred_out]
#grab file names
labels = weir_learn.data.test_dl.dl.dataset.x.items
#output
df = pd.DataFrame({"Filename": labels, "Correct": pred_class})
df.to_csv(out_path + str(type_) + "folder_pred.csv", index = False)
weir_learn.save(out_path + str(type_) + "_learner")
