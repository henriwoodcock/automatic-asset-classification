import os
import pandas as pd
import shutil

types = ["embankment", "flood_gate", "flood_wall", "outfall", "reservoir", "weir"]
cwd = os.getcwd()
csv_path = os.getcwd() + "/data/processing/"
full_image_set = os.getcwd() + "/data/processing/"
output_loc = os.getcwd() + "/data/final_dataset/rough/"

for type_ in types:
    pred_folder_path = csv_path + type_ + "folder_pred.csv"
    pred_folder = pd.read_csv(pred_folder_path)

    for i in range(len(pred_folder)):
        file_name = pred_folder["Filename"][i]
        category = pred_folder["Correct"][i]
        if category != "yes":
            shutil.move(file_name, output_loc + type_ + "/no/" + type_ + "_" + str(i + 1) + ".jpg")
        else:
            shutil.move(file_name, output_loc + type_ + "/yes/" + type_ + "_" + str(i + 1) + ".jpg")

    print(type_, "completed")
