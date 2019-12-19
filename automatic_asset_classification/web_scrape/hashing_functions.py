import hashlib
from hashlib import md5
import imageio
import numpy as np
import scipy
import os
import cv2


def file_hash(filepath):
    with open(filepath, 'rb') as f:
        return md5(f.read()).hexdigest()

def filter_images(images):
    '''
    a function to make all images 3d (which will later be turned into black and white)
    '''
    image_list = []
    for image in images:
        try:
            assert imageio.imread(image).shape[2] == 3 or imageio.imread(image).shape[2] == 4
            image_list.append(image)
        except AssertionError:
            print(image, "does not contain 3channels")
        except ValueError:
            print(image)
        except IndexError:
            print(image, "has one channel")
            os.remove(image)

    return image_list

def img_gray(image):
    '''
    this function turns an image into a gray image
    '''
    image = imageio.imread(image)
    if image.shape[2] == 4:
        image = image[:,:,:3]

    return np.average(image, weights=[0.299, 0.587, 0.114], axis=2)

def resize(image, height=30, width=30):
    '''
    resize and flatten image
    '''
    row_res = cv2.resize(image,(height, width), interpolation = cv2.INTER_AREA).flatten()
    col_res = cv2.resize(image,(height, width), interpolation = cv2.INTER_AREA).flatten('F')
    return row_res, col_res

def intensity_diff(row_res, col_res):
    '''
    calculates differences in intensities and then calculates gradient from
    '''
    difference_row = np.diff(row_res)
    difference_col = np.diff(col_res)
    difference_row = difference_row > 0
    difference_col = difference_col > 0
    return np.vstack((difference_row, difference_col)).flatten()

def difference_score(image, height = 30, width = 30):
    '''
    calculate difference for an image
    '''
    gray = img_gray(image)
    row_res, col_res = resize(gray, height, width)
    difference = intensity_diff(row_res, col_res)

    return difference

def difference_score_dict_hash(image_list):
    '''
    creates a dictionary based on difference score
    '''
    ds_dict = {}
    duplicates = []
    hash_ds = []
    for image in image_list:
        ds = difference_score(image)
        hash_ds.append(ds)
        filehash = md5(ds).hexdigest()
        if filehash not in ds_dict:
            ds_dict[filehash] = image
        else:
            duplicates.append((image, ds_dict[filehash]) )

    return  duplicates, ds_dict, hash_ds

def hamming_distance(image, image2):
    '''
    return hamming distance between two Images
    '''
    score =scipy.spatial.distance.hamming(image, image2)
    return score

def difference_score_dict(image_list):
    '''
    create a dict for similar scores with hamming distance
    '''
    ds_dict = {}
    duplicates = []
    for image in image_list:
        ds = difference_score(image)

        if image not in ds_dict:
            ds_dict[image] = ds
        else:
            duplicates.append((image, ds_dict[image]) )

    return  duplicates, ds_dict
