import tensorflow as tf
import numpy as np
import convnet as cv
from skimage import io as skio
from skimage import transform as sktf
import os
import time
import FISHMONGERv3 as fm

def return_images(directory):
    """Returns all .jpg images in a specified directory
    you know, in UNIX you can just use glob and it's way easier
    Returns the images with their full path names
    """
    allfiles = os.listdir(directory)
    image_list = [im for im in allfiles if '.jpg' in str(im)]
    image_list = [directory + im for im in image_list]
    return image_list

def read_images(image_info, image_dims):
    """"Reads in a list of images, returns a 2D array
    image_info should be a list of tuples, where first entry is the filename, second entry is the class, then num_chan
    Each image is read in, interpolated according to image_dims, normalized, and flattened
    Returns a 2D array where the first entry corresponds to the image class (as given in image_info)
    Note: classes are stored as floats, so they need to be retyped as ints when processing
    """
    num_examples = len(image_info)
    num_pixels = int(image_dims[0]*image_dims[1]*image_dims[2])
    locations, classes = zip(*image_info)
    output_array = np.zeros((num_examples, num_pixels+1), dtype=np.float32)
    for entry in range(num_examples):
        if entry % 100 == 0:
            print('reading image: '+str(entry)+'/'+str(num_examples))
        output_array[entry, 0] = classes[entry]  # image classes
        input_image = skio.imread(locations[entry], as_grey=False)  # read in a color image
        output_image = sktf.resize(input_image, image_dims)  # interpolate down to image_dims (including channels)
        """normalize images by color channel, with fuzz factor to avoid div0"""
        maxpx = np.zeros((1, image_dims[2]))  # store max/min for each channel
        minpx = np.zeros((1, image_dims[2]))
        for i in range(image_dims[2]):  # find max/min for each channel
            maxpx[0, i] = np.max(output_image[:, :, i])
            if maxpx[0, i] == float(0):
                maxpx[0, i] = 1e-12  # fuzz factor
            minpx[0, i] = np.min(output_image[:, :, i])
        """flatten and store"""
        for i in range(image_dims[2]):
            output_array[entry, 1+i*(image_dims[0]*image_dims[1]):1+(i+1)*(image_dims[0]*image_dims[1])] = \
                np.ravel((output_image[:, :, i] - minpx[0, i]) / (maxpx[0, i] - minpx[0, i]))
    return output_array

def batch_create(all_images, image_list, image_dims, num_classes):
    """Creates 4D array with dims [batch_size, x, y, z] to be fed into network,
    and a 2D array with dims [batch_size, num_classes] for one-hot encoded classes
    all_images should be the return of read_images(): n rows, 0th entry is class, 1:i entry px values
    image_list should be a batch-sized iterable of row indices to fetch
    image_dims should be a 3 entry iterable of (x, y, z)
    This should be converted to TensorFlow at some point
    """
    output_array = np.zeros((len(image_list), image_dims[0], image_dims[1], image_dims[2]), dtype=np.float32)
    class_array = np.zeros((len(image_list), num_classes), dtype=np.int8)
    counter = 0  # I don't think I've used a counter since CS150
    for image in image_list:
        pixelvals = all_images[image, 1:]
        img_class = int(all_images[image, 0])
        output_array[counter, :, :, :] = np.reshape(pixelvals, image_dims)
        class_array[counter, img_class] = 1
        counter += 1
    return output_array, class_array


parent_dir = 'C:\\Users\\Donald\\Desktop\\fishes\\code\\fishMonger\\zFF_test\\'
save_dir = 'C:\\Users\\Donald\Desktop\\fishes\\code\\savestate_fishMonger\\'
img_dims = (100, 100, 3)
allfish = return_images(parent_dir)
fish_encoded = list(zip(allfish, [1.0] * len(allfish)))
total_num_images = len(fish_encoded)
all_image_data = read_images(fish_encoded, img_dims)
input_image_array = np.zeros((total_num_images, 100*100*3 + 3), dtype=np.float32)
input_image_array[:, :3] = 0.0
input_image_array[:, 3:] = all_image_data[:, 1:]
calculated_classes = fm.feed_forward_fishmonger(input_image_array, img_dims, 7, save_dir+'save.ckpt')

