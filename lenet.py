import tensorflow as tf
def Lenet(x):
    """
    Input
    x->contains the images in (patch_size,patch_size,3) format
    The LeNet architecture accepts a 32x32xC image as input, where C is the number of color channels.
    Modifying to accept 64x6pip 
    
    Architecture
    Layer 1: Convolutional. The output shape should be 28x28x6.

    Activation. Your choice of activation function.

    Pooling. The output shape should be 14x14x6.

    Layer 2: Convolutional. The output shape should be 10x10x16.

    Activation. Your choice of activation function.

    Pooling. The output shape should be 5x5x16.

    Flatten. Flatten the output shape of the final pooling layer such that it's 1D instead of 3D. The easiest way to do is by using tf.contrib.layers.flatten, which is already imported for you.

    Layer 3: Fully Connected. This should have 120 outputs.

    Activation. Your choice of activation function.

    Layer 4: Fully Connected. This should have 84 outputs.

    Activation. Your choice of activation function.

    Layer 5: Fully Connected (Logits). This should have 10 outputs.
    """
    