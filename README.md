# Fishes2Riches
Code written as part of the Kaggle Nature Conservancies Fisheries Monitoring (https://www.kaggle.com/c/the-nature-conservancy-fisheries-monitoring). 

# Project Description
This was my first Kaggle competition, completed with Michael Reeks ([@mreeks91](https://github.com/mreeks91)). The goal of this competition was to design an algorithm to identify fish in low-resolution webcam-type images and then classify them into categories (Bluefin, Yellowfin, etc.). Our strategy was to split the problem into two parts: identifying fish in an image and then classifying them. To this end, we implemented two convolutional neural networks in Tensorflow. The first, [fishfinder](https://github.com/dleebrown/Fishes2Riches/blob/master/FISHFINDERv7.py), was presented with a subregion of an image and the network returned the probability of a fish being fed in that subregion. The second, [fishmonger](https://github.com/dleebrown/Fishes2Riches/blob/master/FISHMONGERv4.py), was given an image subregion that fishfinder flagged as containing a fish, and this subregion was then classified as a member of one of the fish categories. Both of these networks consisted of several convolutional layers followed by several dense layers. 

The main challenges of this competition were:

1. Limited and unbalanced training set. Some categories of fish only had ~dozens of training examples, while other had many hundreds . Thus, much of our code development was spent on training set augmentation/balancing. We implemented random flips and rotations, as well as various downsampling schemes, to prevent overfitting. We also implemented a balancing function, in order to ensure that during training, the network was not biased towards ignoring fish classes that were not well-represented in the training set. 

2. Computational resources. We did not have a GPU available during this competition, so we were forced to traing using CPUs. This was challenging, especially after we realized our original network architectures described above were too simple to adequately capture the variance in the training images. We implemented various schemes to reduce computational burden, such as aggressive image [downsampling](https://github.com/dleebrown/Fishes2Riches/blob/master/downsampler.py), preprocessing of training examples, but as our networks grew in complexity we quickly outscaled our hardware resources. For example, we implemented deeper neural networks, [deepfinder](https://github.com/dleebrown/Fishes2Riches/blob/master/deepFISHFINDERv1.py) and 
[deepmonger](https://github.com/dleebrown/Fishes2Riches/blob/master/deepFISHMONGERv2.py), but the networks were taking weeks to train. Eventually, we abandoned the neural networks and switched to boosted decision tree classifiers (xgboost), these implementations are found in [fishbooster](https://github.com/dleebrown/Fishes2Riches/blob/master/FISHBOOSTERv2.py) and 
[boostedmonger](https://github.com/dleebrown/Fishes2Riches/blob/master/BOOSTEDMONGERv2.py). The boosted tree implementations returned better results than the original networks in a fraction of the training time, but we were outclassed in the competition by deep neural networks. 

3. Problem complexity. Especially for a first Kaggle competition, this challenge was pretty difficult. The images were low-resolution and noisy, the fish could be photographed from any angle (and were occasionally obscured or missing pieces), and the limited training set size necessitated some creative problem-solving. Our hardware limitations also forced us to think carefully about our implementations and eliminate code bottlenecks. 

# Authors
Donald Lee-Brown ([@dleebrown](https://github.com/dleebrown)), Michael Reeks ([@mreeks91](https://github.com/mreeks91))



