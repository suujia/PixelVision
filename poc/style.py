import os
import sys
import numpy as np
import scipy.io
import scipy.misc
import tensorflow as tf
from PIL import Image

import models

# Neural training proof-of-concept
# Source: http://www.chioka.in/tensorflow-implementation-neural-algorithm-of-artistic-style
# Make sure you download this model: http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat
# and place it in the /poc/images folder

OUTPUT_DIR = './output/'
STYLE_IMAGE = './images/monet.jpg'
CONTENT_IMAGE = './images/susie.jpg'
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 360
COLOR_CHANNELS = 3

# Noise weight to intermix with content image
NOISE_RATIO = 0.6
# Content loss
BETA = 5
# Style loss
ALPHA = 100

# VGG 19-layer model by from the paper "Very Deep Convolutional
# Networks for Large-Scale Image Recognition"
VGG_MODEL = 'imagenet-vgg-verydeep-19.mat'
# Scale VGG model for performance
MEAN_VALUES = np.array([123.68, 116.779, 103.939]).reshape((1, 1, 1, 3))

def content_loss_func(sess, model):
    '''
    Content loss function as defined in the paper.
    '''
    def _content_loss(p, x):
        # N is the number of filters (at layer l).
        N = p.shape[3]
        # M is the height times the width of the feature map (at layer l).
        M = p.shape[1] * p.shape[2]
        # Interestingly, the paper uses this form instead:
        #
        #   0.5 * tf.reduce_sum(tf.pow(x - p, 2))
        #
        # But this form is very slow in "painting" and thus could be missing
        # out some constants (from what I see in other source code), so I'll
        # replicate the same normalization constant as used in style loss.
        return (1 / (4 * N * M)) * tf.reduce_sum(tf.pow(x - p, 2))
    return _content_loss(sess.run(model['conv4_2']), model['conv4_2'])

STYLE_LAYERS = [
    ('conv1_1', 0.5),
    ('conv2_1', 1.0),
    ('conv3_1', 1.5),
    ('conv4_1', 3.0),
    ('conv5_1', 4.0),
]

def style_loss_func(sess, model):
    '''
    Style loss function as defined in the paper.
    '''
    def _gram_matrix(F, N, M):
        '''
        The gram matrix G.
        '''
        Ft = tf.reshape(F, (M, N))
        return tf.matmul(tf.transpose(Ft), Ft)

    def _style_loss(a, x):
        '''
        The style loss calculation.
        '''
        # N is the number of filters (at layer l).
        N = a.shape[3]
        # M is the height times the width of the feature map (at layer l).
        M = a.shape[1] * a.shape[2]
        # A is the style representation of the original image (at layer l).
        A = _gram_matrix(a, N, M)
        # G is the style representation of the generated image (at layer l).
        G = _gram_matrix(x, N, M)
        result = (1 / (4 * N**2 * M**2)) * tf.reduce_sum(tf.pow(G - A, 2))
        return result

    E = [_style_loss(sess.run(model[layer_name]), model[layer_name]) for layer_name, _ in STYLE_LAYERS]
    W = [w for _, w in STYLE_LAYERS]
    loss = sum([W[l] * E[l] for l in range(len(STYLE_LAYERS))])
    return loss

def generate_noise_image(content_image, noise_ratio = NOISE_RATIO):
    '''
    Returns a noise image intermixed with the content image at a certain ratio.
    '''
    noise_image = np.random.uniform(
            -20, 20,
            (1, IMAGE_HEIGHT, IMAGE_WIDTH, COLOR_CHANNELS)).astype('float32')
    # White noise image from the content representation. Take a weighted average
    # of the values
    input_image = noise_image * noise_ratio + content_image * (1 - noise_ratio)
    return input_image

def load_image(path):
    image = scipy.misc.imread(path)
    # Resize the image for convnet input, there is no change but just
    # add an extra dimension.
    image = np.reshape(image, ((1,) + image.shape))
    # Input to the VGG model expects the mean to be subtracted.
    image = image - MEAN_VALUES
    return image

def save_image(path, image):
    # Output should add back the mean.
    image = image + MEAN_VALUES
    # Get rid of the first useless dimension, what remains is the image.
    image = image[0]
    image = np.clip(image, 0, 255).astype('uint8')
    scipy.misc.imsave(path, image)

#############
# THE STUFF #
#############

session = tf.InteractiveSession()
content_image = load_image(CONTENT_IMAGE)
style_image = load_image(STYLE_IMAGE)
model = models.load_vgg_model('./images/imagenet-vgg-verydeep-19.mat', IMAGE_HEIGHT, IMAGE_WIDTH, COLOR_CHANNELS)

# Construct content loss and style loss
session.run(tf.global_variables_initializer())
session.run(model['input'].assign(content_image))
session.run(model['input'].assign(style_image))
content_loss = content_loss_func(session, model)
style_loss = style_loss_func(session, model)
print content_loss
print style_loss

# Equation 7 of paper (see above) - turns out, alpha and beta are just scaling factors
total_loss = (ALPHA * style_loss) + (BETA * content_loss)

# From the paper: jointly minimize the distance of a white noise image
# from the content representation of the photograph in one layer of
# the neywork and the style representation of the painting in a number
# of layers of the CNN.
# The content is built from one layer, while the style is from five
# layers. Then we minimize the total_loss, which is the equation 7.
optimizer = tf.train.AdamOptimizer(2.0)
train_step = optimizer.minimize(total_loss)

input_image = generate_noise_image(content_image)
session.run(tf.global_variables_initializer())
session.run(model['input'].assign(input_image))
for it in range(500):
    session.run(train_step)
    if it%5 == 0:
        # Print every 100 iteration.
        mixed_image = session.run(model['input'])
        print 'Iteration %d' % (it)
        print 'sum : ', session.run(tf.reduce_sum(mixed_image))
        print 'cost: ', session.run(total_loss)

        if not os.path.exists(OUTPUT_DIR):
            os.mkdir(OUTPUT_DIR)

        filename = 'output/%d.png' % (it)
        save_image(filename, mixed_image)
