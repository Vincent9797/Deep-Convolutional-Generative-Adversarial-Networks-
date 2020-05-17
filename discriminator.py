from keras.layers import *
from keras.models import Model
from keras.optimizers import Adam

def get_discriminator(input_shape):
  inputs = Input(input_shape)

  x = Conv2D(64, kernel_size=5, strides=2, padding='same')(inputs)
  x = LeakyReLU(0.2)(x)
  x = Dropout(0.3)(x)

  x = Conv2D(128, kernel_size=5, strides=2, padding='same')(x)
  x = LeakyReLU(0.2)(x)
  x = Dropout(0.3)(x)

  x = Flatten()(x)
  x = Dense(1, activation='sigmoid')(x)

  model = Model(inputs, x)
  model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))
  return model