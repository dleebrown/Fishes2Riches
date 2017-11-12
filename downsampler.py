from skimage import io as skio
from skimage import transform as sktf
from skimage.viewer import ImageViewer
from skimage import util as skut
import random as rd
import numpy as np
import math
import time
import os

"""takes in a bunch of directories, and dumps downsampled images into a different set of identical, yet empty,
directories somwhere. does this by reading in the good old fishclicker giant text file. since some images are host to
multiple cutouts with different sizes, if the image has, e.g., name 00121.jpg with cutouts of size 200 and 300,
will save two images: 00121_200.jpg and 00121_300.jpg. will just overwrite if that size has already been saved.

Will obviously need to modify read_single_image to reflect that filename.
"""

def return_image_names(directory):
    """Returns all .jpg images in a specified directory
    you know, in UNIX you can just use glob and it's way easier
    Returns the images with their full path names
    """
    allfiles = os.listdir(directory)
    image_list = [im for im in allfiles if '.jpg' in str(im)]
    image_list = [directory + im for im in image_list]
    return image_list


def downsample(image_entry, directory, output_directory, output_size):
    """assumes image_entry is the same form as the fishclicker
    thus when throwing these downsampled images through monger/finder, can just use 100x100 cutout sizes and
    the appropriate scaling factors for the cutout locations"""
    image_name = directory+'img_'+image_entry[0][1:-1]+'.jpg'
    output_name = output_directory+'img_'+image_entry[0][1:-1]+'_'+str(image_entry[3])+'.jpg'
    image = skio.imread(image_name, as_grey=False)
    #  scale and downsample the image here to reduce computation
    o_size = np.shape(image)
    scaling = float(output_size[0])/float(image_entry[3])
    o_shape = [int(scaling*o_size[0]), int(scaling*o_size[1])]
    output_image = sktf.resize(image, o_shape)
    skio.imsave(output_name, output_image)


def return_image_info(filename):
    """Reads in a file in a directory, assuming it is the output of the newfangled fish clicking gui,
    i.e. each line in the gui output file looks like ['image_num', [xcenter,ycenter], size, 'Class']
    does the necessary stripping of useless characters and returns a list of tuples, each of form
    ('image_num', xcenter, ycenter, size, 'class')
    """
    openfile = open(filename, 'r')
    imagelist = [line.rstrip(']\n') for line in openfile]
    imagelist = [line.lstrip('[') for line in imagelist]
    outputlist = []
    for line in imagelist:
        line = line.split(',')
        line = [entry.strip() for entry in line]
        line = [entry.rstrip(']') for entry in line]
        line = [entry.lstrip('[') for entry in line]
        outputlist.append((line[0], line[1], line[2], line[3], line[4]))
    return outputlist

if __name__ == "__main__":

    image_entries = return_image_info('C:/Users/Donald/Desktop/fishes/data/train/YFT_cutouts.txt')
    train_imdir = 'C:/Users/Donald/Desktop/fishes/data/train/YFT/'
    out_imdir = 'C:/Users/Donald/Desktop/fishes/data/train_ds/YFT/'
    train_dirs_dict = {'ALB': train_imdir+'ALB/', 'BET': train_imdir+'BET/', 'DOL': train_imdir+'DOL/',
                       'LAG': train_imdir+'LAG/', 'OTHER': train_imdir+'OTHER/', 'SHARK': train_imdir+'SHARK/',
                       'YFT': train_imdir+'YFT/'}
    output_dirs_dict = {'ALB': out_imdir+'ALB/', 'BET': out_imdir+'BET/', 'DOL': out_imdir+'DOL/',
                       'LAG': out_imdir+'LAG/', 'OTHER': out_imdir+'OTHER/', 'SHARK': out_imdir+'SHARK/',
                       'YFT': out_imdir+'YFT/'}
    imsize = [100, 100, 3]
    for i in image_entries:
        downsample(i, train_imdir, out_imdir, imsize)
    """
    input_directory = 'C:/Users/Donald/Desktop/fishes/data/train/NoF/'
    output_directory = 'C:/Users/Donald/Desktop/fishes/data/train_ds/NoF/'
    """

    #  image_names = return_image_names(input_directory)
    """
    for entry in image_names:
        entry = entry.split('/')[-1]
        entry = entry.split('_')[1]
        entry = entry.split('.')[0]
        psuedo_name = "'"+str(entry)+"'"
        image_entry = [psuedo_name, 20, 20, 0, 'NoF']
        for i in [100, 150, 200, 250, 300, 350, 400]:
            image_entry[3] = i
            downsample(image_entry, train_imdir, out_imdir, imsize)
    """





