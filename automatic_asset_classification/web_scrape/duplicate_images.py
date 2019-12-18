import hashlib
from scipy.misc import imread, imresize, imshow
import matplotlib.pyplot as plt
import time
import numpy as np
import os
import imageio
from automatic_asset_classification.web_scrape import hashing_functions
import cv
import itertools

types = ["embankment", "flood_gate", "flood_wall", "outfall", "reservoir", "weir"]

#first check for broken images:
broken_images = dict()
total_broken_images = 0
for type in types:
    #go to folder containing images of type
    loc = os.getcwd() + "/data/raw/" + type
    files_list = os.listdir(loc)

    cant_open = []
    for index, filename in enumerate(files_list):
        #try to open image
        try:
            imageio.imread(loc + "/" + filename)

        except ValueError:
            cant_open.append(filename)
    print(len(cant_open), "broken files in", type)
    broken_images[type] = cant_open
    total_broken_images += len(cant_open)
print(total_broken_images, "broken images in total")


#first find exact duplicates using hashing
total_num_of_dupes = 0
#diction to contain duplicate images for each type
all_dupes = dict()
for type in types:
    #go to folder containing images of type
    loc = os.getcwd() + "/data/raw/" + type
    files_list = os.listdir(loc)

    duplicates = []
    hash_keys = dict()
    for index, filename in  enumerate(files_list):
        #check if file exists
        if os.path.isfile(filename):
            with open(filename, 'rb') as f:
                #produce hash of file
                filehash = hashlib.md5(f.read()).hexdigest()
            #check if file already in hash_keys
            if filehash not in hash_keys:
                hash_keys[filehash] = index
            #if file already in hash_keys add to duplicates
            else:
                duplicates.append((index,hash_keys[filehash]))
    #add duplicates to dictionary
    all_dupes[type] = duplicates
    total_num_of_dupes += len(duplicates)

    print("There are", len(duplicates), "duplicates in", type)

print(total_num_of_dupes, "found in total from simple hashing")


#now try dhashing for similar images
total_num_of_dupes = 0
all_dupes = dict()
for type in types:
    #go to folder containing images of type
    loc = os.getcwd() + "/data/raw/" + type
    files_list = os.listdir(loc)
    files_list = [loc + "/" + file for file in files_list]

    image_files = hashing_functions.filter_images(files_list)
    duplicates, ds_dict, hash_ds = hahsing_functions.difference_score_dict_hash(image_files)
    all_dupes[type] = duplicates

    total_num_of_dupes += len(duplicates)
    print("There are", len(duplicates), "duplicates in", type)

print(total_num_of_dupes, "found in total from dhashing")


#now try hamming distance for similar images:
for k1,k2 in itertools.combinations(ds_dict, 2):
    if hamming_distance(ds_dict[k1], ds_dict[k2])< .10:
        duplicates.append((k1,k2))
