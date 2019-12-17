import hashlib
from scipy.misc import imread, imresize, imshow
import matplotlib.pyplot as plt
import time
import numpy as np
import os
import imageio

def file_hash(filepath):
    with open(filepath, 'rb') as f:
        return md5(f.read()).hexdigest()

loc = os.getcwd() + "/data/raw/embankment"

files_list = os.listdir(loc)
print(len(files_list))

duplicates = []
hash_keys = dict()
for index, filename in  enumerate(os.listdir('.')):  #listdir('.') = current directory
    if os.path.isfile(filename):
        with open(filename, 'rb') as f:
            filehash = hashlib.md5(f.read()).hexdigest()
        if filehash not in hash_keys:
            hash_keys[filehash] = index
        else:
            duplicates.append((index,hash_keys[filehash]))

duplicates



/Users/henriwoodcock/Documents/Code/data_projects/automatic-asset-classification/data/raw/embankment
