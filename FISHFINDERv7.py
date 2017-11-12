"""
see fishMonger_readme
"""
import time
import tensorflow as tf
import numpy as np
import convnet as cv
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
"""
Various helper functions
"""


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


def train_batch_create(imagedirs, classes, indices, image_key, offset_percent, output_size):
    """Creates 4D array with dims [batch_size, x, y, z] to be fed into network,
    and a 2D array with dims [batch_size, num_classes] for one-hot encoded classes
    imagedirs should be the dictionary with directories, classes should be the same with class assignments
    indices should be a list of indices of image_key to pull out for constructing the batch
    offset percent and output size are for read_single_image
    """
    batch_size = len(indices)
    n_classes = len(classes)
    #  now create the output class and pixel arrays
    output_array = np.zeros((batch_size, output_size[0], output_size[1], output_size[2]), dtype=np.float32)
    class_array = np.zeros((batch_size, n_classes), dtype=np.int8)
    for i in range(batch_size): #  reads in the images, applying postprocessing
        current_image = image_key[indices[i]]
        fish_type = current_image[-1]
        fish_type = fish_type.strip("'")
        fish_class = int(classes[fish_type])
        image_data = read_single_image(current_image, imagedirs[fish_type], offset_percent, output_size)
        output_array[i, :, :, :] = np.reshape(image_data, output_size)
        #  assumes classes are 0-n:
        class_array[i, fish_class] = 1
    return output_array, class_array


def batch_worker(minibatch_info, frozen_params):
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
    nclass = len(classes)
    nfish = len(minibatch_info)
    class_onehot = np.zeros((nfish, nclass), dtype=np.int8)
    imdata = np.zeros((nfish, int(np.prod(output_size))))
    for i in range(nfish):
        current_fishtuple = minibatch_info[i]
        fish_type = current_fishtuple[-1]
        fish_type = fish_type.strip("'")
        fish_directory = imagedirs[fish_type]
        imdata[i, :] = read_single_image(current_fishtuple, fish_directory, offset_percent, output_size)
        if fish_type != 'NoF':
            fish_type = 'FISH'
        fish_class = int(classes[fish_type])
        class_onehot[i, fish_class] = 1
    return imdata, class_onehot


def train_batch_create_mp(imagedirs, classes, indices, image_key, offset_percent, output_size, nprocesses):
    """Creates 4D array with dims [batch_size, x, y, z] to be fed into network,
    and a 2D array with dims [batch_size, num_classes] for one-hot encoded classes
    imagedirs should be the dictionary with directories, classes should be the same with class assignments
    indices should be a list of indices of image_key to pull out for constructing the batch
    offset percent and output size are for read_single_image
    """
    batch_size = len(indices)
    n_classes = len(classes)
    #  now create the output class and pixel arrays
    output_array = np.zeros((batch_size, output_size[0], output_size[1], output_size[2]), dtype=np.float32)
    class_array = np.zeros((batch_size, n_classes), dtype=np.int8)
    batch_data = [image_key[i] for i in indices]
    whole_minibatch_size = batch_size // nprocesses
    num_whole_minibatches = batch_size // whole_minibatch_size
    input_list = []
    for i in range(num_whole_minibatches):
        input_list.append(batch_data[whole_minibatch_size*i:whole_minibatch_size*(1+i)])
    if batch_size % nprocesses != 0:
        input_list.append(batch_data[whole_minibatch_size*num_whole_minibatches:])
    frozen_params = (imagedirs, classes, offset_percent, output_size)
    partial_worker = partial(batch_worker, frozen_params=frozen_params)
    # initializes the pool of processes
    print('building pool')
    pool = multiprocessing.Pool(nprocesses)
    # maps partial_worker and list of stars to the pool, stores used parameters in a list
    print('mapping pool')
    outputs = pool.map(partial_worker, input_list)
    # end the list of functions to go to pool
    pool.close()
    print('pool closed')
    # wait for all processes to return
    pool.join()
    print('pool joined')
    counter = 0
    for i in range(len(outputs)):
        current_output = outputs[i]
        pixel_data = current_output[0]
        class_data = current_output[1]
        num_fish = len(pixel_data)
        for lf in range(num_fish):
            output_array[counter, :, :, :] = np.reshape(pixel_data[lf], output_size)
            class_array[counter, :] = class_data[lf]
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


"""
General variables for the network
"""
"""whatever, these are just hardcoded because I need to rework this at some point"""
img_x = 100
img_y = 100
img_z = 1
numb_class = 2

with tf.variable_scope('FISHFINDER'):
    with tf.name_scope('IMAGE_DIMS'):
        # placeholders for image_x, image_y
        im_x = tf.placeholder(tf.int32)
        im_y = tf.placeholder(tf.int32)
        im_z = tf.placeholder(tf.int32)

    with tf.name_scope('NUM_CLASSES'):
        # placeholder for the number of classes
        num_classes = tf.placeholder(tf.int32)

    with tf.name_scope('INPUT_BATCH'):
        # should be said that by the time this is filled, all the images must be the same dimensions 100x100?
        image_data = tf.placeholder(tf.float32, shape=[None, img_x, img_y, img_z])

    with tf.name_scope('KNOWN_CLASS'):
        # is it a fish, or is it not a fish?
        true_class = tf.placeholder(tf.float32, shape=[None, numb_class])

    with tf.name_scope('C1_DIMS'):
        # shape of convolutional layer 1: [n_x, n_y, n_input_chan, n_output_chan]
        c1_shape = [5, 5, img_z, 8]

    with tf.name_scope('C1_POOL'):
        # stride for pooling of first conv layer
        c1_pool = [1, 2, 2, 1]

    with tf.name_scope('C2_DIMS'):
        # shape of convolutional layer 2: [n_x, n_y, n_input_chan, n_output_chan]
        c2_shape = [10, 10, 8, 8]

    with tf.name_scope('C1_POOL'):
        # stride for pooling of second conv layer
        c2_pool = [1, 2, 2, 1]

    with tf.name_scope('F1_WEIGHTS'):
        # number of weights to use in first fully-connected layer
        f1_num_weights = 256

    with tf.name_scope('F2_WEIGHTS'):
        # number of weights to use in second fully-connected layer
        f2_num_weights = 256

    with tf.name_scope('KEEP_PROB'):
        # probability of retaining activation during dropout
        # store as a placeholder in order to turn off during testing
        keep_prob = tf.placeholder(tf.float32)

    """
    Network Architecture
    2 convolutional layers with dropout
    2 fully connected layers with dropout
    relu activations throughout
    1 output layer (outputs logit)
    cross entropy cost (probably switch to their cost function at some point)
    adam optimizer
    """

    with tf.name_scope('conv_1_params'):
        # weight and bias initialization for first convolutional layer
        c1_bias = cv.generate_bias(length=c1_shape[3])
        c1_weight = cv.generate_weights(c1_shape, type='p_uniform')

    with tf.name_scope('conv_1'):
        # first convolutional layer with relu and dropout, max pooling
        c1 = cv.layer_conv(input=image_data, weights=c1_weight)
        c1_activate = cv.layer_activate(input=c1, bias=c1_bias, type='relu')
        c1_dropout = cv.layer_dropout(c1_activate, keep_probability=keep_prob)
        c1_pooled = cv.layer_pool(c1_dropout, stride=c1_pool, type='max')

    with tf.name_scope('conv_2_params'):
        # weight and bias initialization for second convolutional layer
        c2_bias = cv.generate_bias(length=c2_shape[3])
        c2_weight = cv.generate_weights(c2_shape, type='p_uniform')

    with tf.name_scope('conv_2'):
        # second convolutional layer with relu and dropout, max pooling
        c2 = cv.layer_conv(input=c1_pooled, weights=c2_weight)
        c2_activate = cv.layer_activate(input=c2, bias=c2_bias, type='relu')
        c2_dropout = cv.layer_dropout(c2_activate, keep_probability=keep_prob)
        c2_pooled = cv.layer_pool(c2_dropout, stride=c2_pool, type='max')

    with tf.name_scope('flatten'):
        # flatten output of convolutional network
        flattened, total_params = cv.layer_flatten(c2_pooled)

    with tf.name_scope('fc_1_params'):
        # weight and bias initialization for first fully-connected layer
        f1_weight = cv.generate_weights([total_params, f1_num_weights], type='p_uniform')
        f1_bias = cv.generate_bias(f1_num_weights)

    with tf.name_scope('fc_1'):
        # first fully-connected layer with relu and dropout
        f1 = cv.layer_fc(input=flattened, weights=f1_weight)
        f1_activate = cv.layer_activate(input=f1, bias=f1_bias, type='relu')
        f1_dropout = cv.layer_dropout(input=f1_activate, keep_probability=keep_prob)

    with tf.name_scope('fc_2_params'):
        # weight and bias initialization for second fully-connected layer
        f2_weight = cv.generate_weights([f1_num_weights, f2_num_weights], type='p_uniform')
        f2_bias = cv.generate_bias(f2_num_weights)

    with tf.name_scope('fc_2'):
        # second fully-connected layer with relu and dropout
        f2 = cv.layer_fc(input=f1_dropout, weights=f2_weight)
        f2_activate = cv.layer_activate(input=f2, bias=f2_bias, type='relu')
        f2_dropout = cv.layer_dropout(input=f2_activate, keep_probability=keep_prob)

    with tf.name_scope('fc_3_params'):
        # initialize weights for output layer of network
        f3_weight = cv.generate_weights([f2_num_weights, numb_class], type='p_uniform')

    with tf.name_scope('fc_3'):
        # compute output of the network with logit output
        f3 = cv.layer_fc(input=f2_dropout, weights=f3_weight)

    with tf.name_scope('cost'):
        # cost function for the network
        cost = cv.network_cost(predicted_vals=f3, known_vals=true_class, type='x_entropy_soft')

    with tf.name_scope('optimize'):
        # optimization function
        optimize = cv.network_optimize(cost, learn_rate=1e-4, optimizer='adam')

    with tf.name_scope('predicted_class'):
        calculated_value = tf.nn.softmax(f3)
        predicted_class = tf.argmax(calculated_value, dimension=1)


def train_network(num_iterations, multiplier, imagedirs, classes, infodict, balancedict, image_dims, batch_size,
                  save_path):
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
    start_time = time.time()
    tf_runtime = 0.0
    preprocess_time = 0.0
    fishinfo = build_fish_info_dict(infodict)
    nofish_images = return_images(imagedirs['NoF'])
    if num_iterations % multiplier != 0:
        print('warning: illegal num_iterations. should be a multiple of multiplier. dying...')
        sys.exit()
    num_loops = num_iterations // multiplier  # num_loops is the number of times to set up the preprocess queue
    counter = 0
    for i in range(num_loops):
        start_time2 = time.time()
        input_queue = []
        for e in range(multiplier):
            single_batch = batch_balance(fishinfo, balancedict, nofish_images)
            input_queue = input_queue + single_batch
        queue_indices = list(range(len(input_queue)))  # this is just a hacky thing to avoid recoding train_batch_create_mp
        print('beginning preprocess')
        px_values, true_classes = train_batch_create_mp(imagedirs, classes, queue_indices, input_queue, 0.15,
                                                        image_dims, nprocesses=3)
        end_time2 = time.time() - start_time2
        preprocess_time += end_time2
        print('beginning tensorflow')
        for itera in range(multiplier):
            start_index = itera*batch_size
            end_index = batch_size*(itera+1)
            input_px = px_values[start_index:end_index, :, :, :]
            input_class = true_classes[start_index:end_index, :]
            input_dict = {image_data: input_px, true_class: input_class, im_x: image_dims[0], im_y: image_dims[1],
                        im_z: image_dims[2], num_classes: 2, keep_prob: 0.9}
            tf_time_start = time.time()
            session.run(optimize, feed_dict=input_dict)  # actually the training
            counter += 1
            """debug
            weightnames = ['c1_weights', 'c2_weights', 'f1_weights', 'f2_weights', 'f3_weights']
            c1s, c2s, f1s, f2s, f3s, c1a, f1a = session.run([c1_weight, c2_weight, f1_weight, f2_weight, f3_weight,
                                                             c1_activate, f1_activate], feed_dict=input_dict)
            output_vals = [c1s, c2s, f1s, f2s, f3s]
            for stat in range(5):
                print(weightnames[stat]+' '+str(np.max(output_vals[stat])))
            end debug
            """
            tf_runtime = tf_runtime + time.time() - tf_time_start
            if counter % 100 == 0:  # diagnostics printed every 100 iterations
                current_cost = session.run(cost, feed_dict=input_dict)
                print('done with iteration ' + str(counter) + '/'+str(num_iterations)+', cost is: ' + str(current_cost))
                decimal_classes, pred_classes = session.run([calculated_value, predicted_class], feed_dict=input_dict)
                trueclass = list(np.where(input_class == 1)[1])
                correct_predictions = [i == j for i, j in zip(trueclass, pred_classes)]
                total_correct_predictions = sum(correct_predictions)
                accuracy = round((float(total_correct_predictions) / float(batch_size)) * 100, 2)
                print('current batch accuracy is ' + str(accuracy)+"%")
            if counter % 500== 0:
                saved = saver.save(session, save_path+'save.ckpt')
                print('Model Checkpointed')
    total_time = time.time() - start_time
    return total_time, preprocess_time, tf_runtime


def test_network_finder(imagedirs, classes, infodict, balancedict, image_dims):
    """Tests the network. Assuming the number of test images is small, this just reads them all in and
    feeds them all through the network at once, recovering what the network thinks about their classiness
    to keep things indicative of how the network would perform on actual cutouts, this applies postprocessing
    """
    start_time = time.time()
    tf_runtime = 0.0
    preprocess_time = 0.0
    fishinfo = build_fish_info_dict(infodict)
    nofish_images = return_images(imagedirs['NoF'])
    start_time2 = time.time()
    input_queue = batch_balance(fishinfo, balancedict, nofish_images)
    queue_indices = list(range(len(input_queue)))  # this is just a hacky thing to avoid recoding train_batch_create_mp
    num_images = len(queue_indices)
    px_values, true_classes = train_batch_create_mp(imagedirs, classes, queue_indices, input_queue, 0.15, image_dims,
                                                    nprocesses=3)
    end_time2 = time.time() - start_time2
    preprocess_time += end_time2
    input_dict = {image_data: px_values, true_class: true_classes, im_x: image_dims[0], im_y: image_dims[1],
                  im_z: image_dims[2], num_classes: 2, keep_prob: 1.0}
    tf_time_start = time.time()
    """Now, the network isn't being optimized, so instead the calculated raw class values and binary class
    predictions are returned for all the test images"""
    decimal_classes, pred_classes = session.run([calculated_value, predicted_class], feed_dict=input_dict)
    tf_runtime = tf_runtime + time.time() - tf_time_start
    pred_classes = np.ravel(pred_classes)  # munge munge munge
    pred_classes = pred_classes.tolist()  # munge munge munge munge munge
    decimal_classes.tolist()  # munge munge munge munge munge munge munge munge munge munge
    trueclass = list(np.where(true_classes == 1)[1])
    correct_predictions = [i == j for i, j in zip(trueclass, pred_classes)]  # boolean of correct predictions
    lists = [trueclass, pred_classes, decimal_classes]
    names = ['true', 'pred', 'raw']
    for i in range(3):
        """see readme for description of these output files"""
        file = open('ff_out'+names[i]+'.txt', 'w')
        for line in lists[i]:
            file.write(str(line)+' \n')
        file.close()
    total_correct_predictions = sum(correct_predictions)  # total number of correct test predictions
    accuracy = round((float(total_correct_predictions) / float(num_images))*100, 2)
    total_time = time.time() - start_time
    return total_time, tf_runtime, accuracy


def feed_forward_fishfinder(image_array, image_dims, n_classes, saved_state):
    """"Given an array of image data, will load a saved network state, feed the data forward, and return the
    probabilities of each fish class for each image.
    > image_array should be a numpy float32 array with dimensions [num_stars, num_px+3], where for each star,
    the first entry is the x coordinate of the cutout center, the second entry is the y coordinate, and the third entry
    is the pixel scale, i.e. the multiplier that would need to be applied to the number of pixels to recover the
    number of pixels of the cutout in the original, non-down (or up) sampled image. The assumption is that image_array
    contains cutouts from just one image, so there is no space for identifying which image a particular row came from.
    The remaining entries in each row are the pixel values themselves (3 color, (x,y,z))
    > image_dims should be a tuple (or other iterable) with the (x,y,z) dimensions of each image fed(e.g. (100,100,3))
    > num_classes should be the number of classes (7 for just fish/other)
    > saved state should be a path + filename pointing to a saved neural network state trained on images of identical
    dimensions to image_dims. (e.g. "C:\\Users\\Donald\\Desktop\\random_folder\saved_data\save.ckpt")
    This function returns a numpy array with dimensions [num_fish, n], where the entries for each fish are
    [center_x, center_y, px_scale, prob_per_class]
    """
    num_stars = np.size(image_array[:, 0])
    output_array = np.zeros((num_stars, 3+n_classes))  # output array: [cent_x, cent_y, scale, prob_nofish, prob_fish]
    output_array[:, 0:3] = image_array[:, 0:3]
    reshaped_image_array = np.zeros((num_stars, image_dims[0], image_dims[1], image_dims[2]), dtype=np.float32)
    placeholder_class = np.ones((num_stars, n_classes))
    for entry in range(num_stars):
        px_data = image_array[entry, 3:]
        reshaped_image_array[entry, :, :, :] = np.reshape(px_data, image_dims)
    input_dict = {image_data: reshaped_image_array, true_class: placeholder_class,
                  im_x: image_dims[0], im_y: image_dims[1], im_z: image_dims[2],
                  num_classes: n_classes, keep_prob: 1.0}

    session = tf.Session()
    saver_ff = tf.train.Saver([var for var in tf.global_variables() if 'FISHFINDER' in var.name])
    saver_ff.restore(session, saved_state)
    decimal_classes = session.run([calculated_value], feed_dict=input_dict)
    output_classes = decimal_classes[0][:]
    output_array[:, 3:] = output_classes
    session.close()
    return output_array


if __name__ == "__main__":
    """"Here's how I ran the network on my computer.
    If you want to train, flip train to YES. If you want to continue training, flip conintue_train to YES.
    If you want to test, flip test to YES. Flip them to something else (e.g., NO) if you don't want to do one.
    Change the directories to be appropriate."""
    train = 'NO'
    continue_train = 'NO'
    test = 'YES'
    ff_test = 'NO'
    save_dir = 'C:/Users/Donald\Desktop/fishes/code_revamp/savestate_fishFinder/'  # directory to save weights in
    train_imdir = 'C:/Users/Donald/Desktop/fishes/data/train_ds/'  # directory with all training image subdirectories
    test_imdir = 'C:/Users/Donald/Desktop/fishes/data/test_ds/'
    img_dims = (100, 100, 1)
    class_dict = {'FISH': 0, 'NoF': 1}
    train_dirs_dict = {'ALB': train_imdir+'ALB/', 'BET': train_imdir+'BET/', 'DOL': train_imdir+'DOL/',
                       'LAG': train_imdir+'LAG/', 'OTHER': train_imdir+'OTHER/', 'SHARK': train_imdir+'SHARK/',
                       'YFT': train_imdir+'YFT/', 'NoF': train_imdir+'NoF/'}
    test_dirs_dict = {'ALB': test_imdir+'ALB/', 'BET': test_imdir+'BET/', 'DOL': test_imdir+'DOL/',
                      'LAG': test_imdir+'LAG/', 'OTHER': test_imdir+'OTHER/', 'SHARK': test_imdir+'SHARK/',
                      'YFT': test_imdir+'YFT/', 'NoF': test_imdir+'NoF/'}
    train_key_dict = {'ALB': train_imdir+'ALB_cutouts.txt', 'BET': train_imdir+'BET_cutouts.txt',
                      'DOL': train_imdir+'DOL_cutouts.txt', 'LAG': train_imdir+'LAG_cutouts.txt',
                      'OTHER': train_imdir+'OTHER_cutouts.txt', 'SHARK': train_imdir+'SHARK_cutouts.txt',
                      'YFT': train_imdir+'YFT_cutouts.txt'}
    test_key_dict = {'ALB': test_imdir+'ALB_cutouts_t.txt', 'BET': test_imdir+'BET_cutouts_t.txt',
                      'DOL': test_imdir+'DOL_cutouts_t.txt', 'LAG': test_imdir+'LAG_cutouts_t.txt',
                      'OTHER': test_imdir+'OTHER_cutouts_t.txt', 'SHARK': test_imdir+'SHARK_cutouts_t.txt',
                      'YFT': test_imdir+'YFT_cutouts_t.txt'}
    train_bal_dict = {'ALB': 14, 'BET': 5, 'DOL': 5,
                      'LAG': 5, 'OTHER': 7, 'SHARK': 5,
                      'YFT': 9, 'NoF': 50}
    test_bal_dict = {'ALB': 30, 'BET': 10, 'DOL': 5,
                      'LAG': 5, 'OTHER': 15, 'SHARK': 10,
                      'YFT': 25, 'NoF': 100}
    if train == 'YES':
        print('beginning training...')
        session = tf.Session()
        session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        t1, t2, t3 = train_network(num_iterations=22000, multiplier=100, imagedirs=train_dirs_dict, classes=class_dict,
                                   infodict=train_key_dict, balancedict=train_bal_dict, image_dims=img_dims,
                                   batch_size=100, save_path=save_dir)
        save_path = saver.save(session, save_dir+'save.ckpt')
        print('Model saved: ' + save_dir)
        session.close()
        print('total run: ' + str(round(t1, 2)))
        print('preprocess time: ' + str(round(t2, 2)))
        print('tensorflow run time: ' + str(round(t3, 2)))

    if continue_train == 'YES':
        #  Allows for loading of previously saved session for more training
        session = tf.Session()
        saver = tf.train.Saver()
        saver.restore(session, save_dir+'save.ckpt')
        print('Model restored, continuing training...')
        t1, t2, t3 = train_network(num_iterations=22000, multiplier=100, imagedirs=train_dirs_dict, classes=class_dict,
                                   infodict=train_key_dict, balancedict=train_bal_dict, image_dims=img_dims,
                                   batch_size=100, save_path=save_dir)
        save_path = saver.save(session, save_dir + 'save.ckpt')
        print('Model saved: ' + save_path)
        session.close()
        print('total run time: ' + str(round(t1, 2)))
        print('preprocess time: ' + str(round(t2, 2)))
        print('tensorflow run time: ' + str(round(t3, 2)))

    if test == 'YES':
        session = tf.Session()
        saver_ff = tf.train.Saver([var for var in tf.global_variables() if 'FISHFINDER' in var.name])
        saver_ff.restore(session, save_dir+'save.ckpt')
        print('Model restored, testing CNN')
        t1, t2, acc = test_network_finder(test_dirs_dict, class_dict, test_key_dict, test_bal_dict, img_dims)
        session.close()
        print('total run time: ' + str(round(t1, 2)))
        print('tensorflow run time: ' + str(round(t2, 2)))
        print('classification accuracy: '+str(acc)+'%')

    if ff_test == 'YES':
        from skimage import transform as sktf
        parent_dir = 'C:\\Users\\Donald\\Desktop\\fishes\\code\\fishMonger\\zFF_test\\'
        img_dims = (100, 100, 3)
        allfish = return_images(parent_dir)
        fish_encoded = list(zip(allfish, [1.0] * len(allfish)))
        total_num_images = len(fish_encoded)
        all_image_data = read_images(fish_encoded, img_dims)
        input_image_array = np.zeros((total_num_images, 100 * 100 * 3 + 3), dtype=np.float32)
        input_image_array[:, :3] = 0.0
        input_image_array[:, 3:] = all_image_data[:, 1:]
        calculated_classes = feed_forward_fishfinder(input_image_array, img_dims, 7, save_dir + 'save.ckpt')
        print(calculated_classes)
