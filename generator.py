from keras.layers import *
from keras.models import Model
from keras.optimizers import Adam

def get_generator(input_shape, final_dim=(28,28), initial_channels=128, final_channels=1): # final_dim in (H,W)
  inputs = Input(input_shape)
  x = Dense(initial_channels*(final_dim[0]//4)*(final_dim[1]//4))(inputs)

  x = LeakyReLU(0.2)(x)
  x = Reshape(target_shape=(final_dim[0]//4,final_dim[1]//4,initial_channels))(x)
  x = UpSampling2D(size=(2,2))(x)

  x = Conv2D(64, kernel_size=5, padding='same')(x)
  x = LeakyReLU(0.2)(x)
  x = UpSampling2D(size=(2,2))(x)

  x = Conv2D(final_channels, kernel_size=5, padding='same', activation='tanh')(x)

  model = Model(inputs, x)
  model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))
  return model