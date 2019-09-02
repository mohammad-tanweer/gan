# Introduction 
Deep Convolution GAN or DCGAN in short:
In this notebook, you'll build a GAN using convolutional layers in the generator and discriminator. This is called a Deep Convolutional GAN, or DCGAN for short. The DCGAN architecture was first explored last year and has seen impressive results in generating new images, you can read the [original paper here](https://arxiv.org/pdf/1511.06434.pdf).

Training DCGAN on the [Street View House Numbers](http://ufldl.stanford.edu/housenumbers/) (SVHN) dataset. These are color images of house numbers collected from Google street view. SVHN images are in color and much more variable than MNIST. 

![SVHN Examples](./../assets/SVHN_examples.png)

So, we'll need a deeper and more powerful network. This is accomplished through using convolutional layers in the discriminator and generator. It's also necessary to use batch normalization to get the convolutional networks to train.

Learnt this from the [Generative Adversarial Networks (GANs) by Udacity - The Complete Youtube Playlist](https://mc.ai/generative-adversarial-networks-gans-by-udacity-the-complete-youtube-playlist/)
# Getting Started

## Software Dependencies
* Python 3.6
* TensorFlow 1.14.0


### User Guide