import pickle as pkl

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
import tensorflow as tf

# Load the data. The SVHN dataset is in matlab format, but we cse read this data in python using scipy's loadmat
dataDir = "./../data/"

trainSet = loadmat(dataDir + "train_32x32.mat")
testSet = loadmat(dataDir + "test_32x32.mat")

'''
We need to do a bit of preprocessing and getting the images into a form where we 
can pass batches to the network. First off, we need to rescale the images to a 
range of -1 to 1, since the output of our generator is also in that range. 
We also have a set of test and validation images which could be used if we're 
trying to identify the numbers in the images.
'''


def scale(x, featureRange=(-1, 1)):
    # scale to (0, 1)
    x = ((x - x.min()) / (255 - x.min()))

    # scale to feature_range
    min, max = featureRange
    x = x * (max - min) + min
    return x


class Dataset:
    def __init__(self, train, test, valFrac=0.5, shuffle=False, scaleFunc=None):
        splitIdx = int(len(test['y']) * (1 - valFrac))
        self.testX, self.validX = test['X'][:, :, :, :splitIdx], test['X'][:, :, :, splitIdx:]
        self.testy, self.validy = test['y'][:splitIdx], test['y'][splitIdx:]
        self.trainX, self.trainy = train['X'], train['y']

        self.trainX = np.rollaxis(self.trainX, 3)
        self.validX = np.rollaxis(self.validX, 3)
        self.testX = np.rollaxis(self.testX, 3)

        if scaleFunc is None:
            self.scaler = scale
        else:
            self.scaler = scaleFunc
        self.shuffle = shuffle

    def batches(self, batchSize):
        if self.shuffle:
            idx = np.arange(len(self.trainX))
            np.random.shuffle(idx)
            self.trainX = self.trainX[idx]
            self.trainy = self.trainy[idx]

        nBatches = len(self.trainy) // batchSize
        for ii in range(0, len(self.trainy), batchSize):
            x = self.trainX[ii:ii + batchSize]
            y = self.trainy[ii:ii + batchSize]

            yield self.scaler(x), y

    def valid(self):
        x = self.validX
        y = self.validy

        return self.scaler(x), y


def model_inputs(realDim, zDim):
    inputsReal = tf.placeholder(tf.float32, (None, *realDim), name='inputReal')
    inputsZ = tf.placeholder(tf.float32, (None, zDim), name='inputZ')

    return inputsReal, inputsZ


def generator(z, outputDim, reuse=False, alpha=0.2, training=True):
    with tf.variable_scope('generator', reuse=reuse):
        # First fully connected layer
        x1 = tf.layers.dense(z, 4 * 4 * 512)
        # Reshape it to start the convolutional stack
        x1 = tf.reshape(x1, (-1, 4, 4, 512))
        x1 = tf.layers.batch_normalization(x1, training=training)
        x1 = tf.maximum(alpha * x1, x1)
        # 4x4x512 now

        x2 = tf.layers.conv2d_transpose(x1, 256, 5, strides=2, padding='same')
        x2 = tf.layers.batch_normalization(x2, training=training)
        x2 = tf.maximum(alpha * x2, x2)
        # 8x8x256 now

        x3 = tf.layers.conv2d_transpose(x2, 128, 5, strides=2, padding='same')
        x3 = tf.layers.batch_normalization(x3, training=training)
        x3 = tf.maximum(alpha * x3, x3)
        # 16x16x128 now

        # Output layer
        logits = tf.layers.conv2d_transpose(x3, outputDim, 5, strides=2, padding='same')
        # 32x32x3 now

        out = tf.tanh(logits)

        return out


def discriminator(x, reuse=False, alpha=0.2):
    with tf.variable_scope('discriminator', reuse=reuse):
        # Input layer is 32x32x3
        x1 = tf.layers.conv2d(x, 64, 5, strides=2, padding='same')
        relu1 = tf.maximum(alpha * x1, x1)
        # 16x16x64

        x2 = tf.layers.conv2d(relu1, 128, 5, strides=2, padding='same')
        bn2 = tf.layers.batch_normalization(x2, training=True)
        relu2 = tf.maximum(alpha * bn2, bn2)
        # 8x8x128

        x3 = tf.layers.conv2d(relu2, 256, 5, strides=2, padding='same')
        bn3 = tf.layers.batch_normalization(x3, training=True)
        relu3 = tf.maximum(alpha * bn3, bn3)
        # 4x4x256

        # Flatten it
        flat = tf.reshape(relu3, (-1, 4 * 4 * 256))
        logits = tf.layers.dense(flat, 1)
        out = tf.sigmoid(logits)

        return out, logits


def model_loss(inputReal, inputZ, outputDim, alpha=0.2):
    """
    Get the loss for the discriminator and generator
    :param input_real: Images from the real dataset
    :param input_z: Z input
    :param out_channel_dim: The number of channels in the output image
    :return: A tuple of (discriminator loss, generator loss)
    """
    gModel = generator(inputZ, outputDim, alpha=alpha)
    dModelReal, dLogitsReal = discriminator(inputReal, alpha=alpha)
    dModelFake, dLogitsFake = discriminator(gModel, reuse=True, alpha=alpha)

    dLossReal = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=dLogitsReal, labels=tf.ones_like(dModelReal)))
    dLossFake = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=dLogitsFake, labels=tf.zeros_like(dModelFake)))
    gLoss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=dLogitsFake, labels=tf.ones_like(dModelFake)))

    dLoss = dLossReal + dLossFake

    return dLoss, gLoss

def model_opt(dLoss, gLoss, learningRate, beta1):
    """
    Get optimization operations
    :param d_loss: Discriminator loss Tensor
    :param g_loss: Generator loss Tensor
    :param learning_rate: Learning Rate Placeholder
    :param beta1: The exponential decay rate for the 1st moment in the optimizer
    :return: A tuple of (discriminator training operation, generator training operation)
    """
    # Get weights and bias to update
    tVars = tf.trainable_variables()
    dVars = [var for var in tVars if var.name.startswith('discriminator')]
    gVars = [var for var in tVars if var.name.startswith('generator')]

    # Optimize
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        dTrainOpt = tf.train.AdamOptimizer(learningRate, beta1=beta1).minimize(dLoss, var_list=dVars)
        gTrainOpt = tf.train.AdamOptimizer(learningRate, beta1=beta1).minimize(gLoss, var_list=gVars)

    return dTrainOpt, gTrainOpt


class GAN:
    def __init__(self, realSize, zSize, learningRate, alpha=0.2, beta1=0.5):
        tf.reset_default_graph()

        self.inputReal, self.inputZ = model_inputs(realSize, zSize)

        self.dLoss, self.gLoss = model_loss(self.inputReal, self.inputZ,
                                              realSize[2], alpha=alpha)

        self.dOpt, self.gOpt = model_opt(self.dLoss, self.gLoss, learningRate, beta1)


def train(net, dataset, epochs, batchSize, printEvery=10, showEvery=100, figsize=(5, 5)):
    saver = tf.train.Saver()
    sampleZ = np.random.uniform(-1, 1, size=(72, zSize))

    samples, losses = [], []
    steps = 0

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for e in range(epochs):
            for x, y in dataset.batches(batchSize):
                steps += 1

                # Sample random noise for G
                batchZ = np.random.uniform(-1, 1, size=(batchSize, zSize))

                # Run optimizers
                _ = sess.run(net.dOpt, feed_dict={net.inputReal: x, net.inputZ: batchZ})
                _ = sess.run(net.gOpt, feed_dict={net.inputZ: batchZ, net.inputReal: x})

                if steps % printEvery == 0:
                    # At the end of each epoch, get the losses and print them out
                    trainLossD = net.dLoss.eval({net.inputZ: batchZ, net.inputReal: x})
                    trainLossG = net.gLoss.eval({net.inputZ: batchZ})

                    xValid, yValid = dataset.valid()
                    validLossD = net.dLoss.eval({net.inputZ: batchZ, net.inputReal: xValid})

                    print("Epoch {}/{}...".format(e + 1, epochs),
                          "Discriminator Loss: {:.4f}...".format(trainLossD),
                          "Generator Loss: {:.4f}".format(trainLossG),
                          "Discriminator Loss - Valid: {:.4f}...".format(validLossD))
                    # Save losses to view after training
                    losses.append((trainLossD, trainLossG))

                # if steps % show_every == 0:
                #     gen_samples = sess.run(
                #         generator(net.input_z, 3, reuse=True, training=False),
                #         feed_dict={net.input_z: sample_z})
                #     samples.append(gen_samples)
                #     _ = view_samples(-1, samples, 6, 12, figsize=figsize)
                #     plt.show()

        saver.save(sess, './checkpoints/generator.ckpt')

    with open('samples.pkl', 'wb') as f:
        pkl.dump(samples, f)

    return losses, samples


realSize = (32,32,3)
zSize = 100
learningRate = 0.0002
batchSize = 128
epochs = 25
alpha = 0.2
beta1 = 0.5

# Create the network
net = GAN(realSize, zSize, learningRate, alpha=alpha, beta1=beta1)

dataset = Dataset(trainSet, testSet)

losses, samples = train(net, dataset, epochs, batchSize, figsize=(10,5))