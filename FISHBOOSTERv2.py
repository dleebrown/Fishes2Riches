import numpy as np
import time
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import metrics as skm
from sklearn.model_selection import RandomizedSearchCV
from skimage import io as skio
import os
import random as rd
import math
# from skimage.viewer import ImageViewer
from skimage import util as skut
import multiprocessing
from functools import partial
import sys
from PIL import Image
import pickle


def return_images(directory):
    """Returns all .jpg images in a specified directory
    you know, in UNIX you can just use glob and it's way easier
    Returns the images with their full path names
    """
    allfiles = os.listdir(directory)
    image_list = [im for im in allfiles if '.jpg' in str(im)]
    image_list = [directory + im for im in image_list]
    return image_list


def return_image_info(filename):
    """Reads in a file, assuming it is the output of the newfangled fish clicking gui,
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


def image_normalize(image, image_dims):
    """min/max normalization with fuzz factor
    image should be an array-like skio image
    image_dims should be a 3-tuple [x,y,z]
    recasts image as float32 if needed
    """
    if image.dtype != 'float32':
        image = image.astype(dtype=np.float32)
    if np.size(image_dims) == 2:
        maxpx = np.max(image[:, :])
        if maxpx == float(0):
            maxpx = 1e-12  # fuzz factor
        minpx = np.min(image[:, :])
        image[:, :] = (image[:, :] - minpx) / (maxpx - minpx)
    else:
        for i in range(image_dims[2]):  # find max/min for each channel
            maxpx = np.max(image[:, :, i])
            if maxpx == float(0):
                maxpx = 1e-12  # fuzz factor
            minpx = np.min(image[:, :, i])
            image[:, :, i] = (image[:, :, i] - minpx) / (maxpx - minpx)
    return image


def image_pad(image, pixel_loc_x, pixel_loc_y):
    """pads an image with 100 pixels on each side of black (0.0) for each channel
    also corrects pixel_loc_x and pixel_loc_y appropriately to reflect the padding
    returns the padded image, pixel_loc_x, and pixel_loc_y"""
    input_size = np.shape(image)
    padded_image = np.zeros((input_size[0]+200, input_size[1]+200, 1))
    if np.size(input_size) == 2:
            padded_image[:, :, 0] = skut.pad(image[:, :], 100, mode='constant', constant_values=float(0))
    else:
        for i in range(input_size[2]):
            if i == 0:
                padded_image[:, :, 0] = skut.pad(image[:, :, i], 100, mode='constant', constant_values=float(0))
            else:
                padded_dim = np.zeros((input_size[0]+200, input_size[1]+200, 1))
                padded_dim[:, :, 0] = skut.pad(image[:, :, i], 100, mode='constant', constant_values=float(0))
                padded_image = np.append(padded_image, padded_dim, axis=2)
    pixel_loc_x = pixel_loc_x + 100
    pixel_loc_y = pixel_loc_y + 100
    return padded_image, pixel_loc_x, pixel_loc_y


def read_single_image(image_entry, dir, offset_percent, output_size, normalize=True, rotate=True, preds=True):
    """Reads in image and returns a downsampled cutout as a flattened (1D) np array
    classes should be the same with class assignments
    Image entry should be a tuple ('image_num', xcenter, ycenter, size, 'class')
    dir should be the directory the image can be found in (assumes image name is 'img_<image_entry[0]>.jpg')
    Offset percent should be the maximum percent a cutout should have its center offset (20% = 0.20)
    (for simplicity, randomly draw a percent up to offset_percent for each of x and y independently)
    output_size is the output image size, given as a tuple. Should be a 3-tuple: for grayscale images, 3rd entry = 1
    If normalize true, then the cutout is normed (before rotation, after translation) using minmax rotation.
    If rotate is true, will apply random rotation to the image (rotates corner px, then takes all pixels within
    that rotated set of 4 corners) before downsampling for output. The rotated image is interped onto 2d array of size
    specified in the cutout.
    If preds is true, assumes images have been pre-downsampled to appropriate cutout size and uses image names as
    output by downsampler. If not true, then downsamples the images to appropriate size.
    """
    if preds:
        image_name = dir+'img_'+image_entry[0][1:-1]+'_'+str(image_entry[3])+'.jpg'
        if output_size[2] == 1:
            full_image = skio.imread(image_name, as_grey=True)  # read in a greyscale
        else:
            full_image = skio.imread(image_name, as_grey=False)
        scaling = float(output_size[0])/float(image_entry[3])
    else:
        from skimage import transform as sktf
        image_name = dir + 'img_' + image_entry[0][1:-1] + '.jpg'
        if output_size[2] == 1:
            full_image = skio.imread(image_name, as_grey=True)  # read in a greyscale
        else:
            full_image = skio.imread(image_name, as_grey=False)
        #  scale and downsample the image here to reduce computation
        o_size = np.shape(full_image)
        scaling = float(output_size[0])/float(image_entry[3])
        o_shape = [int(scaling*o_size[0]), int(scaling*o_size[1])]
        full_image = sktf.resize(full_image, o_shape)
    image_size = np.shape(full_image)
    if normalize:
        full_image = image_normalize(full_image, image_size)  # normalizes the image that was read in
    else:
        full_image = full_image
    #  compute random center offsets
    cent_x = float(image_entry[1]) + float(image_entry[3])*rd.uniform(-1.0*float(offset_percent), float(offset_percent))
    cent_y = float(image_entry[2]) + float(image_entry[3])*rd.uniform(-1.0*float(offset_percent), float(offset_percent))
    #  compute a corner of the image cutout to use as starting point for making matrix of cutout coordinates
    left_x = scaling*(cent_x - 0.5 * float(image_entry[3]))
    top_y = scaling*(cent_y - 0.5 * float(image_entry[3]))
    pixel_locations_x = np.zeros(output_size[0:2])  # create a 2D array to hold all the pixel locations of the cutout
    pixel_locations_y = np.zeros(output_size[0:2])
    for i in range(output_size[0]):  # leverage the fact that along an axis, x/y locations are identical
        pixel_locations_x[:, i] = left_x + i*1.0
        pixel_locations_y[i, :] = top_y + i*1.0
    #  ravel them to make easier to process
    pixel_locations_x = np.ravel(pixel_locations_x)
    pixel_locations_y = np.ravel(pixel_locations_y)
    if rotate:
        angle = rd.uniform(0.0, 6.284)  # select a random rotation angle
        sina = math.sin(angle)
        cosa = math.cos(angle)
        rotmat = np.array([[cosa, -1.0 * sina], [sina, cosa]])
        #  the rotation should occur about the center of the image cutout location, so translate the origin:
        rel_loc_x = [i - scaling*cent_x for i in pixel_locations_x]
        rel_loc_y = [i - scaling*cent_y for i in pixel_locations_y]
        #  rotate the corners now
        for i in range(len(pixel_locations_x)):
            rotated_coord = np.matmul(rotmat, np.array([[rel_loc_x[i]], [rel_loc_y[i]]]))
            pixel_locations_x[i] = rotated_coord[0, 0] + scaling*cent_x
            pixel_locations_y[i] = rotated_coord[1, 0] + scaling*cent_y
    #  now go ahead and use the rotated (or unrotated, if rotate=false) corners to actually extract the image cutout
    #  first round corners to be the nearest integer
    pixel_locations_x = np.array([int(i) for i in pixel_locations_x])
    pixel_locations_y = np.array([int(i) for i in pixel_locations_y])
    #  if the computed pixel locations are outside the bounds of the image, pad the image with black
    if (np.min(pixel_locations_x)<=0 or np.min(pixel_locations_y) <= 0
        or np.max(pixel_locations_x) >= image_size[1] or np.max(pixel_locations_y) >= image_size[0]):
        full_image, pixel_locations_x, pixel_locations_y = image_pad(full_image, pixel_locations_x, pixel_locations_y)
    """debug
    print('x_cent '+str(scaling*cent_x))
    print('y_cent '+str(scaling*cent_y))
    viewer = ImageViewer(full_image)
    viewer.show()
    """
    output_image = np.ravel(full_image[pixel_locations_y, pixel_locations_x])
    """
    output_image = np.reshape(full_image[pixel_locations_y, pixel_locations_x], output_size)
    viewer2 = ImageViewer(output_image)
    viewer2.show()
    """
    return output_image


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
        input_image = skio.imread(locations[entry], as_grey=False)  # read in a grayscale image
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


def train_batch_create_xgb(imagedirs, classes, indices, image_key, offset_percent, output_size):
    """Creates 2D array with dims [batch_size, px] to be fed into xgboost,
    and a 1D array with dims [batch_size] for class designation
    imagedirs should be the dictionary with directories, classes should be the same with class assignments
    indices should be a list of indices of image_key to pull out for constructing the batch
    offset percent and output size are for read_single_image
    """
    batch_size = len(indices)
    #  now create the output class and pixel arrays
    output_array = np.zeros((batch_size, output_size[0]*output_size[1]*output_size[2]), dtype=np.float32)
    class_array = np.zeros((batch_size), dtype=np.int8)
    for i in range(batch_size): #  reads in the images, applying postprocessing
        current_image = image_key[indices[i]]
        fish_type = current_image[-1]
        fish_type = fish_type.strip("'")
        fish_class = int(classes[fish_type])
        image_data = read_single_image(current_image, imagedirs[fish_type], offset_percent, output_size)
        output_array[i, :] = image_data
        #  assumes classes are 0-n:
        class_array[i] = fish_class
    return output_array, class_array


def batch_worker_xgb(minibatch_info, frozen_params):
    """This is just a wrapper function in order to use multiprocessing to speed up the batch assembly
    Returns the image data (raveled) and the onehot encoded class information
    Frozen params is a tuple containing (imagedirs, classes, offset_percent, output_size) to allow for use of partial
    minibatch_info is a tuple of tuples, each subtuple contains the standard fishclicker format info
    loops over those and returns an array containing imdata and class_onehot to concat with other batch workers
    """
    imagedirs = frozen_params[0]
    classes = frozen_params[1]
    offset_percent = frozen_params[2]
    output_size = frozen_params[3]
    nfish = len(minibatch_info)
    class_array = np.zeros(nfish, dtype=np.int8)
    imdata = np.zeros((nfish, int(np.prod(output_size))))
    for i in range(nfish):
        current_fishtuple = minibatch_info[i]
        fish_type = current_fishtuple[-1]
        fish_type = fish_type.strip("'")
        fish_directory = imagedirs[fish_type]
        imdata[i, :] = read_single_image(current_fishtuple, fish_directory, offset_percent, output_size, rotate=True)
        fish_class = int(classes[fish_type])
        if fish_class != 0:
            fish_class = 1
        class_array[i] = fish_class
    return imdata, class_array


def train_batch_create_mp_xgb(imagedirs, classes, indices, image_key, offset_percent, output_size, nprocesses):
    """Creates 2D array with dims [batch_size, pixels] to be fed into xgboost,
    and a 1D array with dim [batch_size] for numerical class
    imagedirs should be the dictionary with directories, classes should be the same with class assignments
    indices should be a list of indices of image_key to pull out for constructing the batch
    offset percent and output size are for read_single_image
    """
    batch_size = len(indices)
    n_classes = len(classes)
    #  now create the output class and pixel arrays
    output_array = np.zeros((batch_size, output_size[0]*output_size[1]*output_size[2]), dtype=np.float32)
    class_array = np.zeros(batch_size, dtype=np.int8)
    batch_data = [image_key[i] for i in indices]
    whole_minibatch_size = batch_size // nprocesses
    num_whole_minibatches = batch_size // whole_minibatch_size
    input_list = []
    for i in range(num_whole_minibatches):
        input_list.append(batch_data[whole_minibatch_size*i:whole_minibatch_size*(1+i)])
    if batch_size % nprocesses != 0:
        input_list.append(batch_data[whole_minibatch_size*num_whole_minibatches:])
    frozen_params = (imagedirs, classes, offset_percent, output_size)
    partial_worker = partial(batch_worker_xgb, frozen_params=frozen_params)
    # initializes the pool of processes
    pool = multiprocessing.Pool(nprocesses)
    # maps partial_worker and list of stars to the pool, stores used parameters in a list
    outputs = pool.map(partial_worker, input_list)
    # end the list of functions to go to pool
    pool.close()
    # wait for all processes to return
    pool.join()
    counter = 0
    for i in range(len(outputs)):
        current_output = outputs[i]
        pixel_data = current_output[0]
        class_data = current_output[1]
        num_fish = len(pixel_data)
        for lf in range(num_fish):
            output_array[counter, :] = pixel_data[lf]
            class_array[counter] = class_data[lf]
            counter += 1
    return output_array, class_array


def build_fish_info_dict(cutouts_dict):
    """given a dict of fishclicker cutout results txt files per class return dict w fishclicker outs for each class"""
    keys = list(cutouts_dict.keys())
    output_dict = dict(cutouts_dict)
    for entry in keys:
        output_dict[entry] = return_image_info(cutouts_dict[entry])
    return output_dict


def construct_nofish(num_nofish, nofish_images):
    """takes in the number of nofish cutouts needed and the list of all nofish images,
    returns a list of fishclicker-like tuples with random cutout locations"""
    total_num_nofish = len(nofish_images)
    output_list = []
    for i in range(num_nofish):
        index = rd.randint(0, total_num_nofish-1)
        # use PIL here because it lazy loads the image so it's much faster than skio which reads into memory
        current_im = Image.open(nofish_images[index])
        imwidth, imheight = current_im.size
        y_loc = rd.randint(50, imheight-50)
        x_loc = rd.randint(50, imwidth-50)
        x_loc = str(x_loc)
        y_loc = str(y_loc)
        temp_string = nofish_images[index]
        temp_string = temp_string.split('/')[-1]
        temp_string = temp_string.split('.')[0]
        temp_string = temp_string.split('_')
        cutout_size = temp_string[2]
        img_name = '"'+temp_string[1]+'"'
        output_list.append((str(img_name), x_loc, y_loc, cutout_size, 'NoF'))
    return output_list


def batch_balance(fish_info_dict, balance_dict, nofish_images):
    keys = list(balance_dict.keys())
    batch_out = []
    for fish in keys:
        if fish != 'NoF':
            current_fish_info = fish_info_dict[fish]
            num_fish = len(current_fish_info)
            num_draws = balance_dict[fish]
            indices = [rd.randint(0, num_fish-1) for it in range(num_draws)]
            fish_information = [current_fish_info[e] for e in indices]
            batch_out = batch_out + fish_information
        else:
            num_nofish = balance_dict['NoF']
            nofish_information = construct_nofish(num_nofish, nofish_images)
            batch_out = batch_out + nofish_information
    rd.shuffle(batch_out)
    return batch_out


def tune_model_params(fixed_params, grid_params, cv_folds, image_dims, classes, xval_imagedirs,
                      xval_infodict, xval_balancedict):
    # fix_params and hyperparams are just placeholders to help keep track of current values
    print('tuning parameters...')
    xval_time_start = time.time()
    xval_fishinfo = build_fish_info_dict(xval_infodict)
    xval_nofishinfo = return_images(xval_imagedirs['NoF'])
    xval_batch = batch_balance(xval_fishinfo, xval_balancedict, xval_nofishinfo)
    xval_indices = list(range(len(xval_batch)))
    xval_px, xval_class = train_batch_create_mp_xgb(xval_imagedirs, classes, xval_indices, xval_batch,
                                                    0.15, image_dims, nprocesses=4)
    xval_build_time = time.time() - xval_time_start
    print('finished constructing xval set in: '+str(round(xval_build_time, 3))+' seconds')
    print('cross validating...')
    xval_xgb_start = time.time()
    xval_model = RandomizedSearchCV(estimator=XGBClassifier(**fixed_params), param_distributions=grid_params,
                                    n_iter=6, scoring='f1_macro', n_jobs=4, iid=False, cv=cv_folds)
    xval_model.fit(xval_px, xval_class)
    xval_xgb_time = time.time() - xval_xgb_start
    print('xgb cross validation finished in: '+str(round(xval_xgb_time, 2))+' seconds')
    print('tuned params:')
    print(xval_model.best_params_)


def train_booster(xgb_params, num_rebuilds, num_estimators, image_dims, classes, train_imagedirs, train_infodict,
                  train_balancedict, test_imagedirs, test_infodict, test_balancedict, output_name):
    """Trains the network. imagedirs should be a DICTIONARY, structured as
    {'ALB': albdir, 'BET': betdir...} where the keys are the fish classes and dirs are their training image locations.
    Classes should be similar: dictionary with fish names keyed to their numerical class (1-7 here)
    imagekey should be the output of fishclicker, which contains all the different classes in one file.
    image_dims should be the desired image dimensions [x, y, z].
    Uses random shuffles of imagekey to determine which images to call up for each batch.
    through tensorflow. Re-shuffles as images are re-used to train the network.
    multiplier controls the queue size for multiprocessing the minibatches - batch_size * multiplier = number of
    fish to feed into the parallelized image preprocessing loop (recommend quite a few to reduce overhead, e.g. 100)
    for now, num_iterations must be a multiple of multiplier or the code will be angry
    """
    print('training xgb model...')
    preprocess_time = 0.0
    xgb_time = 0.0
    # first build the test DMatrix, which remains constant over the training process
    test_start_time = time.time()
    test_fishinfo = build_fish_info_dict(test_infodict)
    test_nofishinfo = return_images(test_imagedirs['NoF'])
    test_batch = batch_balance(test_fishinfo, test_balancedict, test_nofishinfo)
    test_indices = list(range(len(test_batch)))
    test_px, test_class = train_batch_create_mp_xgb(test_imagedirs, classes, test_indices, test_batch,
                                                    0.15, image_dims, nprocesses=4)
    test_data = xgb.DMatrix(test_px, label=test_class)
    test_build_time = time.time() - test_start_time
    print('finished constructing test set')
    # now train the xgboost model, using parameters that were previously found through grid searching
    fishinfo = build_fish_info_dict(train_infodict)
    nofish_images = return_images(train_imagedirs['NoF'])
    for i in range(num_rebuilds):  # num rebuilds is the number of redraws of the training set
        preprocess_start = time.time()
        current_batch = batch_balance(fishinfo, train_balancedict, nofish_images)
        batch_indices = list(range(len(current_batch)))
        px_values, true_classes = train_batch_create_mp_xgb(train_imagedirs, classes, batch_indices, current_batch,
                                                            0.15, image_dims, nprocesses=4)
        training_data = xgb.DMatrix(px_values, label=true_classes)
        watchlist = [(training_data, 'train'), (test_data, 'test')]
        preprocess_time += (time.time() - preprocess_start)
        xgb_start = time.time()
        if i == 0:  # initializes the trees and performs the first training loop
            xgb_model = xgb.train(xgb_params, training_data, num_estimators, evals=watchlist, early_stopping_rounds=10)
        else:  # on subsequent batches, continues training the model that was already started
            num_estimators += i*1
            xgb_model = xgb.train(xgb_params, training_data, num_estimators, evals=watchlist,
                                  xgb_model=xgb_model, early_stopping_rounds=10)
        xgb_time += (time.time() - xgb_start)
        if i % 10 == 0:
            print('done with iteration '+str(i)+', saving model')
            with open(str(output_name), 'wb') as fl:  # use pickle because xgb saver doesn't save best iteration
                pickle.dump(xgb_model, fl)
    with open(str(output_name), 'wb') as fl:  # use pickle because xgb saver doesn't save best iteration
        pickle.dump(xgb_model, fl)
    return test_build_time, preprocess_time, xgb_time


def feed_forward_fishbooster(image_array, saved_state, n_classes=2):
    """"Given an array of image data, will load a saved xgboost model, feed the data forward, and return the
    probabilities of each fish class for each image.
    > image_array should be a numpy float32 array with dimensions [num_stars, num_px+3], where for each star,
    the first entry is the x coordinate of the cutout center, the second entry is the y coordinate, and the third entry
    is the pixel scale, i.e. the multiplier that would need to be applied to the number of pixels to recover the
    number of pixels of the cutout in the original, non-down (or up) sampled image. The assumption is that image_array
    contains cutouts from just one image, so there is no space for identifying which image a particular row came from.
    The remaining entries in each row are the pixel values themselves (1 color, (x,y,1)) - these values MUST be 
    normalized in exactly the method used in read_single_image, and must be 10000px total (100x100x1)).
    > saved state should be a path + filename pointing to a saved xgboost model
    > n_classes should be 2 (0 = nofish, 1 = fish)
    This function returns a numpy array with dimensions [num_fish, n], where the entries for each fish are
    [center_x, center_y, px_scale, prob_per_class]
    # probability entries will correspond to the dictionary:
    class_dict = {'NoF': 0, 'FISH':1}
    """
    num_fish = np.size(image_array[:, 0])
    output_array = np.zeros((num_fish, 3+n_classes))  # output array: [cent_x, cent_y, scale, prob_nofish, prob_fish]
    output_array[:, 0:3] = image_array[:, 0:3]
    input_px = xgb.DMatrix(image_array[:, 3:])
    xgb_model = pickle.load(open(saved_state, 'rb'))  # it's a pickle load because xgb saver doesn't save best iter
    output_classes = xgb_model.predict(input_px, ntree_limit=xgb_model.best_ntree_limit)
    #output_classes = xgb_model.predict(input_px)
    # binary classification:
    output_array[:, 4] = output_classes
    output_array[:, 3] = 1.0 - output_classes
    return output_array


if __name__ == "__main__":
    rd.seed(5)
    class_dict = {'NoF': 0, 'ALB': 1, 'BET': 2, 'DOL': 3, 'LAG': 4, 'OTHER': 5, 'SHARK': 6, 'YFT': 7}
    train_imdir = '/home/donald/Desktop/fishes/data/train_ds/'
    test_imdir = '/home/donald/Desktop/fishes/data/test_ds/'
    train_dirs_dict = {'ALB': train_imdir+'ALB/', 'BET': train_imdir+'BET/', 'DOL': train_imdir+'DOL/',
                       'LAG': train_imdir+'LAG/', 'OTHER': train_imdir+'OTHER/', 'SHARK': train_imdir+'SHARK/',
                       'YFT': train_imdir+'YFT/', 'NoF': train_imdir+'NoF/'}
    train_key_dict = {'ALB': train_imdir+'ALB_cutouts.txt', 'BET': train_imdir+'BET_cutouts.txt',
                      'DOL': train_imdir+'DOL_cutouts.txt', 'LAG': train_imdir+'LAG_cutouts.txt',
                      'OTHER': train_imdir+'OTHER_cutouts.txt', 'SHARK': train_imdir+'SHARK_cutouts.txt',
                      'YFT': train_imdir+'YFT_cutouts.txt'}
    train_bal_dict = {'ALB': 3000, 'BET': 500, 'DOL': 400,
                      'LAG': 400, 'OTHER': 500, 'SHARK': 400,
                      'YFT': 1200, 'NoF': 7000}
    test_dirs_dict = {'ALB': test_imdir+'ALB/', 'BET': test_imdir+'BET/', 'DOL': test_imdir+'DOL/',
                      'LAG': test_imdir+'LAG/', 'OTHER': test_imdir+'OTHER/', 'SHARK': test_imdir+'SHARK/',
                      'YFT': test_imdir+'YFT/', 'NoF': test_imdir+'NoF/'}
    test_key_dict = {'ALB': test_imdir+'ALB_cutouts_t.txt', 'BET': test_imdir+'BET_cutouts_t.txt',
                      'DOL': test_imdir+'DOL_cutouts_t.txt', 'LAG': test_imdir+'LAG_cutouts_t.txt',
                      'OTHER': test_imdir+'OTHER_cutouts_t.txt', 'SHARK': test_imdir+'SHARK_cutouts_t.txt',
                      'YFT': test_imdir+'YFT_cutouts_t.txt'}
    test_bal_dict = {'ALB': 30, 'BET': 10, 'DOL': 5,
                      'LAG': 5, 'OTHER': 15, 'SHARK': 10,
                      'YFT': 25, 'NoF': 100}
    img_dims = (100, 100, 1)
    model_name = 'trained_xgb_model'

    """
    # WARNING: the scikit learn wrapper for some unfathomable reason uses aliases for some of the xgb params,
    # namely xgb(num_boost_rounds)=n_estimators, xgb(eta)=learning_rate, xgb(silent)=silent(bool)
    # these are fixed parameters for xgboost: FOR GRIDSEARCHCV F1 SCORE TO WORK NEED SOFTMAX OBJECTIVE
    # ALSO NEED TO SPECIFY A NUM_CLASS IF USING SOFTPROB OBJECTIVE
    hyperparams = {'silent': True, 'nthread': 4, 'objective': 'multi:softmax', 'n_estimators': 250}
    # these are potentially tuneable parameters
    fix_params = {'learning_rate': 0.1, 'scale_pos_weight': 1.0}
    hyperparams.update(fix_params)
    xval_params = {'max_depth': [13, 15, 17], 'min_child_weight': np.linspace(0.2, 0.6, 4),
                   'gamma': np.linspace(0.08, 0.12, 5), 'subsample': np.linspace(0.2, 1.0, 4),
                   'colsample_bytree': np.linspace(0.8, 1.0, 5)}
    tune_model_params(fixed_params=hyperparams, grid_params=xval_params, cv_folds=3, image_dims=img_dims,
                      classes=class_dict, xval_imagedirs=train_dirs_dict, xval_infodict=train_key_dict,
                      xval_balancedict=train_bal_dict)
    """
    """
    # these are tuned through cross-validation
    hyperparams = {'silent': 1, 'nthread': 4, 'objective': 'binary:logistic',
                     'eval_metric': 'logloss'}
    tuned_params = {'max_depth': 7, 'min_child_weight': 0.3, 'gamma': 0.1, 'eta': 0.05, 'subsample': 0.9,
                    'colsample_bytree': 0.9, 'scale_pos_weight': 1.0}
    hyperparams.update(tuned_params)

    t1, t2, t3 = train_booster(xgb_params=hyperparams, num_rebuilds=1, num_estimators=250, image_dims=img_dims,
                               classes=class_dict, train_imagedirs=train_dirs_dict, train_infodict=train_key_dict,
                               train_balancedict=train_bal_dict, test_imagedirs=test_dirs_dict,
                               test_infodict=test_key_dict, test_balancedict=test_bal_dict, output_name=model_name)
    print('done training')
    print('xval_build_time: '+str(round(t1, 3)))
    print('preprocess_time: '+str(round(t2, 3)))
    print('xgb_time: '+str(round(t3, 3)))
    """

    # testing the feedforward function
    test_fishinfo = build_fish_info_dict(test_key_dict)
    test_nofishinfo = return_images(test_dirs_dict['NoF'])
    test_batch = batch_balance(test_fishinfo, test_bal_dict, test_nofishinfo)
    test_indices = list(range(len(test_batch)))
    test_px, test_class = train_batch_create_mp_xgb(test_dirs_dict, class_dict, test_indices, test_batch,
                                                    0.15, img_dims, nprocesses=4)

    test_shape = np.shape(test_px)
    test_input_array = np.zeros((test_shape[0], test_shape[1]+3))
    test_input_array[:, 3:] = test_px[:, :]
    test_outputs = feed_forward_fishbooster(test_input_array, 'fishbooster_model2')
    calculated_class = np.argmax(test_outputs[:, 3:], axis=1)
    total_class = np.size(calculated_class)
    correct_class = 0
    classdiff = test_class - calculated_class
    for i in range(total_class):
        if classdiff[i] == 0:
            correct_class += 1
    print(test_outputs[30:36, 3:])
    print(test_class[30:36])
    print('accuracy: '+str(float(correct_class)/float(total_class)))




