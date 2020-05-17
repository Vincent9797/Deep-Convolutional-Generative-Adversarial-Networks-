import matplotlib.pyplot as plt

# Plot the loss from each batch
def plotLoss(epoch, dLosses, gLosses):
    plt.figure(figsize=(10, 8))
    plt.plot(dLosses, label='Discriminitive loss')
    plt.plot(gLosses, label='Generative loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('dcgan_loss_epoch_%d.png' % epoch)

# Create a wall of generated MNIST images
def plotGeneratedImages(epoch, LATENT_SPACE_DIM, examples=100, dim=(10, 10), figsize=(10, 10)):
    noise = np.random.normal(0, 1, size=[examples, LATENT_SPACE_DIM])
    generatedImages = generator.predict(noise)

    plt.figure(figsize=figsize)
    for i in range(generatedImages.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generatedImages[i, :, :, 0], interpolation='nearest', cmap='gray_r')
        plt.axis('off')
    plt.tight_layout()
    plt.show()
    plt.savefig('images/dcgan_generated_image_epoch_%d.png' % epoch)

def saveModels(epoch, generator, discriminator):
    generator.save('gen_model/dcgan_generator_epoch_%d.h5' % epoch)
    discriminator.save('dis_model/dcgan_discriminator_epoch_%d.h5' % epoch)