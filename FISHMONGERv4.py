"""
Fish Monger v0.1 "sladki payk, domashnaya robota" 50% off BOGO $20 MIR cashback bonus bucks"

see fishMonger_readme
"""

import tensorflow as tf
import numpy as np
import convnet as cv
from skimage import io as skio
from skimage import transform as sktf
import os
import random as rd
import time

import csv

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

"""Old function when images were read in one at a time
def image_munge(image_name, image_dims):
    # Takes in the name of an image, and desired output size,
    # scales the image, grayscales, and returns array of pixel values
    input_image = skio.imread(image_name, as_grey=True)
    output_image = sktf.resize(input_image, image_dims)
    # linearly normalize the image to [0, 1]
    maxpx = np.max(output_image)
    if np.max(output_image) == float(0):  # fuzz parameter to avoid div0
        maxpx = 1e-12
    output_image = (output_image - np.min(output_image))/(maxpx-np.min(output_image))
    return output_image
"""


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


def binary_reader(binary_file):
    """Faster alternative to reading in images here.
    Reads in a binary file of form [num_images, im_x, im_y, num_channels, class1, c1px1.1, c1px1.2...c2px1.1,
    c2px1.2...class2, c1px2.1, c1px2.2...]
    and returns a tuple (img_x, img_y, num_chan) and a 2D array with rows [class, px1.1, px1.2...px2.1...]
    The binary file is memmapped to avoid memory errors
    """
    floatarray = np.memmap(binary_file, dtype=np.float32, mode='r')
    num_images = int(floatarray[0])
    image_dims = (int(floatarray[1]), int(floatarray[2]), int(floatarray[3]))
    num_pix = int(floatarray[1])*int(floatarray[2])*int(floatarray[3])
    output_array = np.zeros((num_images, num_pix+1), dtype=np.float32)
    for i in range(num_images):
        image_start = 4+i*(num_pix+1)
        output_array[i, :] = floatarray[image_start: image_start+num_pix+1]
    del floatarray
    return image_dims, num_images, output_array


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

"""
General variables for the network
"""
"""whatever, these are just hardcoded because I need to rework this at some point"""
img_x = 100
img_y = 100
img_z = 3
numb_class = 7

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


def train_network(num_iterations, batch_size, dirs, image_dims, binary_file='NONE'):
    """Trains the network. Reads in images, shuffles them, builds mini-batches, and feeds
    through tensorflow. Re-shuffles as images are re-used to train the network.
    Technically it doesn't reshuffle the actual images, it shuffles the indices used to reference the images.
    This is faster in terms of the shuffle operation, but slower because of accessing non-contiguous memory
    (the giant image array is still in order in RAM, and you index random parts of it).
    I don't think our dataset is big enough for the latter's price to outweigh the former's benefit.
    """
    start_time = time.time()
    python_runtime = 0
    tf_runtime = 0
    if binary_file == 'NONE':
        """If no binary file specified, then goes and fetches images, interpolates them, and
        returns a giant array containing all the images (see read_images). THIS IS RAM INTENSIVE
        """
        n_classes = len(dirs)  # assumes number of classes is the same as number of input directories
        master_list_encoded = []
        for i in range(len(dirs)):
            imagenames = return_images(dirs[i])
            image_encoded = list(zip(imagenames, [i] * len(imagenames)))
            master_list_encoded += image_encoded
        total_num_images = len(master_list_encoded)
        all_image_data = read_images(master_list_encoded, image_dims)
    else:
        """Alternately, if binary file specified, uses binary_reader to assemble data array"""
        image_dims, total_num_images, all_image_data = binary_reader(binary_file)
        n_classes = max(all_image_data[:, 0]) + 1  # +1 for pesky numbering from 0
    image_indices = list(range(total_num_images))
    rd.shuffle(image_indices)  # shuffles the images
    if total_num_images < batch_size:
        print("warning: batch size is greater than number of training images, default to equal batch size")
        batch_size = total_num_images
    counter = 0
    read_time = (time.time() - start_time)
    for i in range(num_iterations):
        """Main training loop. Builds a mini-batch, then trains the network one step, then repeats. If all images
        have been used in a mini-batch, will re-shuffle the images and start building mini-batches again."""
        start_time2 = time.time()
        if total_num_images / (batch_size*(counter+1)) >= 1.0:  # just keeps track of whether or not all images used
            sublist = image_indices[counter*batch_size:batch_size*(counter+1)]
            px_values, true_classes = batch_create(all_image_data, sublist, image_dims, n_classes)  # see batch_create
            input_dict = {image_data: px_values, true_class: true_classes,
                          im_x: image_dims[0], im_y: image_dims[1], im_z: image_dims[2],
                          num_classes: n_classes, keep_prob: 0.9}
            python_runtime = python_runtime + time.time() - start_time2
            tf_time_start = time.time()
            session.run(optimize, feed_dict=input_dict)  # actually the training
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
            if i % 100 == 0:  # diagnostics printed every 100 iterations
                current_cost = session.run(cost, feed_dict=input_dict)
                print('done with iteration ' + str(i) + '/'+str(num_iterations)+', cost is: ' + str(current_cost))
                decimal_classes, pred_classes = session.run([calculated_value, predicted_class], feed_dict=input_dict)
                trueclass = list(all_image_data[sublist, 0].astype(int))
                correct_predictions = [i == j for i, j in zip(trueclass, pred_classes)]
                total_correct_predictions = sum(correct_predictions)
                accuracy = round((float(total_correct_predictions) / float(batch_size)) * 100, 2)
                print('current batch accuracy is ' + str(accuracy)+"%")
            counter += 1
        else:  # if all images used, re-shuffle the indices and start re-using images
            counter = 0
            rd.shuffle(image_indices)
            python_runtime = python_runtime + time.time() - start_time2
    return read_time, python_runtime, tf_runtime


def test_network_monger(dirs, image_dims):
    """Tests the network. Assuming the number of test images is small, this just reads them all in and
    feeds them all through the network at once, recovering what the network thinks about their classiness"""
    start_time = time.time()
    read_write_time = 0.0
    n_classes = len(dirs)  # assumes number of classes is the same as number of input directories
    master_list_encoded = []
    for i in range(len(dirs)):
        imagenames = return_images(dirs[i])
        image_encoded = list(zip(imagenames, [i] * len(imagenames)))
        master_list_encoded += image_encoded
    total_num_images = len(master_list_encoded)
    all_image_data = read_images(master_list_encoded, image_dims)
    image_indices = list(range(total_num_images))
    px_values, true_classes = batch_create(all_image_data, image_indices, image_dims, n_classes)  # see batch_create
    input_dict = {image_data: px_values, true_class: true_classes,
                  im_x: image_dims[0], im_y: image_dims[1], im_z: image_dims[2],
                  num_classes: n_classes, keep_prob: 1.0}
    read_write_time = read_write_time + time.time() - start_time
    run_time_start = time.time()
    """Now, the network isn't being optimized, so instead the calculated raw class values and binary class
    predictions are returned for all the test images"""
    decimal_classes, pred_classes = session.run([calculated_value, predicted_class], feed_dict=input_dict)
    run_time = time.time() - run_time_start
    current_time = time.time()
    pred_classes = np.ravel(pred_classes)  # munge munge munge
    pred_classes = pred_classes.tolist()  # munge munge munge munge munge
    decimal_classes.tolist()  # munge munge munge munge munge munge munge munge munge munge
    all_test_classes = list(all_image_data[:, 0].astype(int))
    correct_predictions = [i == j for i, j in zip(all_test_classes, pred_classes)]  # boolean of correct predictions
    lists = [all_test_classes, pred_classes, decimal_classes]
    names = ['true', 'pred', 'raw']
    for i in range(3):
        """see readme for description of these output files"""
        file = open('out'+names[i]+'.txt', 'w')
        for line in lists[i]:
            file.write(str(line)+' \n')
        file.close()
    total_correct_predictions = sum(correct_predictions)  # total number of correct test predictions
    accuracy = round((float(total_correct_predictions) / float(total_num_images))*100, 2)
    read_write_time = read_write_time + time.time() - current_time
    return read_write_time, run_time, accuracy


def feed_forward_fishmonger(image_array, image_dims, n_classes, saved_state):
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
    saver = tf.train.Saver()
    saver.restore(session, saved_state)
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
    test = 'NO'
    ff_test = 'YES'
    save_dir = 'C:\\Users\\Donald\Desktop\\fishes\\code\\savestate_fishMonger\\'
    img_dims = (100, 100, 3)
    if train == 'YES':
        parent_dir = 'C:\\Users\\Donald\\Desktop\\fishes\data\\augmented_data\\'
        dirs = [parent_dir + 'ALBaugment\\', parent_dir + 'BETaugment\\', parent_dir + 'DOLaugment\\',
                parent_dir + 'LAGaugment\\',
                parent_dir + 'OTHERaugment\\', parent_dir + 'SHARKaugment\\', parent_dir + 'YFTaugment\\']
        session = tf.Session()
        session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        t1, t2, t3 = train_network(num_iterations=16500, batch_size=100, dirs=dirs,
                                   image_dims=img_dims, binary_file='all_images_numbered_class')
        save_path = saver.save(session, save_dir+'save.ckpt')
        print('Model saved: ' + save_path)
        session.close()
        print('total read time: ' + str(round(t1, 2)))
        print('total python run time: ' + str(round(t2, 2)))
        print('total tensorflow run time: ' + str(round(t3, 2)))

    if continue_train == 'YES':
        """Allows for loading of previously saved session for more training"""
        parent_dir = 'C:\\Users\\Donald\\Desktop\\fishes\data\\augmented_data\\'
        dirs = [parent_dir + 'ALBaugment\\', parent_dir + 'BETaugment\\', parent_dir + 'DOLaugment\\',
                parent_dir + 'LAGaugment\\', parent_dir + 'OTHERaugment\\', parent_dir + 'SHARKaugment\\',
                parent_dir + 'YFTaugment\\']
        session = tf.Session()
        saver = tf.train.Saver()
        saver.restore(session, save_dir+'save.ckpt')
        print('Model restored, continuing training...')
        t1, t2, t3 = train_network(num_iterations=18000, batch_size=100, dirs=dirs, image_dims=img_dims,
                                   binary_file='all_images_numbered_class')
        save_path = saver.save(session, save_dir + 'save.ckpt')
        print('Model saved: ' + save_path)
        session.close()
        print('total read time: ' + str(round(t1, 2)))
        print('total python run time: ' + str(round(t2, 2)))
        print('total tensorflow run time: ' + str(round(t3, 2)))

    if test == 'YES':
        parent_dir = 'C:\\Users\\Donald\\Desktop\\fishes\data\\augmented_test\\'
        dirs = [parent_dir + 'ALBaugment\\', parent_dir + 'BETaugment\\', parent_dir + 'DOLaugment\\',
                parent_dir + 'LAGaugment\\',
                parent_dir + 'OTHERaugment\\', parent_dir + 'SHARKaugment\\', parent_dir + 'YFTaugment\\']
        session = tf.Session()
        saver = tf.train.Saver()
        saver.restore(session, save_dir+'save.ckpt')
        print('Model restored, testing CNN')
        t1, t2, acc = test_network_monger(dirs, image_dims=img_dims)
        session.close()
        print('total test read/write time: ' + str(round(t1, 2)))
        print('total test run time: ' + str(round(t2, 2)))
        print('classification accuracy: '+str(acc)+'%')

    if ff_test == 'YES':
        parent_dir = 'C:\\Users\\Donald\\Desktop\\fishes\\code\\fishMonger\\zFF_test\\'
        save_dir = 'C:\\Users\\Donald\Desktop\\fishes\\code\\savestate_fishMonger\\'
        img_dims = (100, 100, 3)
        allfish = return_images(parent_dir)
        fish_encoded = list(zip(allfish, [1.0] * len(allfish)))
        total_num_images = len(fish_encoded)
        all_image_data = read_images(fish_encoded, img_dims)
        input_image_array = np.zeros((total_num_images, 100 * 100 * 3 + 3), dtype=np.float32)
        input_image_array[:, :3] = 0.0
        input_image_array[:, 3:] = all_image_data[:, 1:]
        calculated_classes = feed_forward_fishmonger(input_image_array, img_dims, 7, save_dir + 'save.ckpt')
