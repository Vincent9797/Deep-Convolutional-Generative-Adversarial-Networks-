import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from keras.datasets import fashion_mnist
from keras.layers import *
from keras.models import Model
from keras.optimizers import Adam

from discriminator import get_discriminator
from generator import get_generator

from utils import plotLoss, plotGeneratedImages, saveModels

# dimension of latent space
LATENT_SPACE_DIM = 100

# Load MNIST data
(X_train, Y_train), (X_test, Y_test) = fashion_mnist.load_data()
X_train = (X_train.astype(np.float32))/255.0
X_train = np.expand_dims(X_train, axis=-1) #(num, H, W, C)

dLosses = []
gLosses = []

# Combined network
generator = get_generator((LATENT_SPACE_DIM,))
discriminator = get_discriminator((28,28,1)) # shape of mnist images

discriminator.trainable = False
ganInput = Input(shape=(LATENT_SPACE_DIM,))
x = generator(ganInput)
ganOutput = discriminator(x)
gan = Model(inputs=ganInput, outputs=ganOutput)
gan.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))

def train(epochs=1, batchSize=128):
    batchCount = X_train.shape[0] // batchSize
    print ('[INIT] Epochs:', epochs)
    print ('[INIT] Batch size:', batchSize)
    print ('[INIT] Batches per epoch:', batchCount)

    for e in range(1, epochs+1):
        print ('-'*15, 'Epoch %d' % e, '-'*15)
        for _ in tqdm(range(batchCount)):
            # Get a random set of input noise and images
            noise = np.random.normal(0, 1, size=[batchSize, LATENT_SPACE_DIM])
            imageBatch = X_train[np.random.randint(0, X_train.shape[0], size=batchSize)]

            # Generate fake MNIST images
            generatedImages = generator.predict(noise)
            X = np.concatenate([imageBatch, generatedImages])

            # Labels for generated and real data
            yDis = np.zeros(2*batchSize)
            # One-sided label smoothing
            yDis[:batchSize] = 0.9

            # Train discriminator
            discriminator.trainable = True
            dloss = discriminator.train_on_batch(X, yDis)

            # Train generator
            noise = np.random.normal(0, 1, size=[batchSize, LATENT_SPACE_DIM])
            yGen = np.ones(batchSize)
            discriminator.trainable = False
            gloss = gan.train_on_batch(noise, yGen)

        # Store loss of most recent batch from this epoch
        dLosses.append(dloss)
        gLosses.append(gloss)

        if e == 1 or e % 5 == 0:
            plotGeneratedImages(e, LATENT_SPACE_DIM)
            saveModels(e, generator, discriminator)

    # Plot losses from every epoch
    plotLoss(e, dLosses, gLosses)

if __name__=='__main__':
    train(50, 128)