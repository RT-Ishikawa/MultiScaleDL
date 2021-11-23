## Definition of the network
## developed by Ryohtaroh T. Ishikawa
## Reference: Ishikawa et al. (2021) accepted to A&A
import numpy as np
import keras
from keras.models import Model
from keras.layers import Dense, Activation
from keras.layers import Conv3D, Conv2D
from keras.layers import Reshape, Input
from keras.layers import GlobalAveragePooling2D, Multiply
from keras.layers.normalization import BatchNormalization
## python version     3.5.2
## keras version      2.3.1

# utilize 3 consective frames
ndt = 3
Nx = 128
Ny = 128
# size of the kernels
Nf1 = 60
Nc1 = 3
Nf2 = 40
Nc2 = 7
Nf3 = 20
Nc3 = 15
Nf4 = 10
Nc4 = 31
Nf5 = 5
Nc5 = 51

## Model description
inputs = Input(shape=image_input_shape)
## First layer
x1 = Conv3D(Nf1, (Nc1,Nc1,ndt), padding='same',kernel_initializer='he_normal')(inputs)
x1 = BatchNormalization()(x1)
x1 = Activation('relu')(x1)
x12 = Conv3D(Nf2, (Nc2,Nc2,ndt), padding='same',kernel_initializer='he_normal')(inputs)
x12 = BatchNormalization()(x12)
x12 = Activation('relu')(x12)
x13 = Conv3D(Nf3, (Nc3,Nc3,ndt), padding='same',kernel_initializer='he_normal')(inputs)
x13 = BatchNormalization()(x13)
x13 = Activation('relu')(x13)
x14 = Conv3D(Nf4, (Nc4,Nc4,ndt), padding='same',kernel_initializer='he_normal')(inputs)
x14 = BatchNormalization()(x14)
x14 = Activation('relu')(x14)
x15 = Conv3D(Nf5, (Nc5,Nc5,ndt), padding='same',kernel_initializer='he_normal')(inputs)
x15 = BatchNormalization()(x15)
x15 = Activation('relu')(x15)
x3 = keras.layers.concatenate([inputs,x1,x12,x13,x14,x15],axis=-1)
x0 = Reshape((Nx,Ny,-1))(x3)
## SE block
xse = GlobalAveragePooling2D()(x0)
xse = Dense(30, activation='relu')(xse)
xse = Dense(411, activation='sigmoid')(xse)
x0 = Multiply()([x0,xse])
## Second layer
x71 = Conv2D(20, (1,1), padding='same',kernel_initializer='he_normal')(x0)
x71 = BatchNormalization()(x71)
x72 = Conv2D(10, (1,1), padding='same',kernel_initializer='he_normal')(x0)
x72 = BatchNormalization()(x72)
x73 = Conv2D(5, (1,1), padding='same',kernel_initializer='he_normal')(x0)
x73 = BatchNormalization()(x73)
x74 = Conv2D(5, (1,1), padding='same',kernel_initializer='he_normal')(x0)
x74 = BatchNormalization()(x74)
x75 = Conv2D(2, (1,1), padding='same',kernel_initializer='he_normal')(x0)
x75 = BatchNormalization()(x75)
x1 = Conv2D(Nf1, (Nc1,Nc1), padding='same',kernel_initializer='he_normal')(x71)
x1 = BatchNormalization()(x1)
x1 = Activation('relu')(x1)
x12 = Conv2D(Nf2, (Nc2,Nc2), padding='same',kernel_initializer='he_normal')(x72)
x12 = BatchNormalization()(x12)
x12 = Activation('relu')(x12)
x13 = Conv2D(Nf3, (Nc3,Nc3), padding='same',kernel_initializer='he_normal')(x73)
x13 = BatchNormalization()(x13)
x13 = Activation('relu')(x13)
x14 = Conv2D(Nf4, (Nc4,Nc4), padding='same',kernel_initializer='he_normal')(x74)
x14 = BatchNormalization()(x14)
x14 = Activation('relu')(x14)
x15 = Conv2D(Nf5, (Nc5,Nc5), padding='same',kernel_initializer='he_normal')(x75)
x15 = BatchNormalization()(x15)
x15 = Activation('relu')(x15)
x0 = keras.layers.concatenate([x0,x1,x12,x13,x14,x15],axis=-1)
## Output
main_output = Conv2D(1,(1,1), activation='linear',kernel_initializer='he_normal')(x0)
mode0 = Model(inputs=inputs,outputs=main_output)


#END
