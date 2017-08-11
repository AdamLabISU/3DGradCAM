import keras
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils, generic_utils
from keras.optimizers import SGD, RMSprop
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape, Layer
from keras.layers.convolutional import Convolution3D, MaxPooling3D, UpSampling3D, ZeroPadding3D, AveragePooling3D
from keras.layers.noise import GaussianNoise
from keras.layers.advanced_activations import LeakyReLU, PReLU, ELU
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l1, l2

def loadModel(model_no,channels,activation,voxel_count,nb_classes):
    # Deep network architechture building
    print('... building the model')
    if model_no == 1 or model_no == 2 or model_no == 3:
        # CNN Hyperparameters
        nb_filters = [8, 8, 16]
        nb_pool = 2
        if model_no == 1:
            conv_size1 = 8
            conv_size2 = 4
        elif model_no == 2:
            conv_size1 = 12
            conv_size2 = 6
        elif model_no == 3:
            conv_size1 = 14
            conv_size2 = 8
            conv_size3 = 4

        model = Sequential()
        # if padding:
        #     model.add(ZeroPadding3D(padding=(2, 2, 2), data_format='channels_last', input_shape=(voxel_count, voxel_count, voxel_count, channels)))
        #     model.add(BatchNormalization(epsilon=1e-05, axis=1, momentum=0.99, weights=None, beta_initializer='zeros', gamma_initializer='ones'))

#  First Convolution3D block
        model.add(Convolution3D(nb_filters[0], conv_size1, padding='valid',data_format='channels_last', input_shape=(voxel_count, voxel_count, voxel_count, channels)))
        if activation == 'relu':
            model.add(Activation('relu'))
        if activation == 'leakyrelu':
            model.add(LeakyReLU(alpha=0.01))
        if activation == 'elu':
            model.add(ELU(alpha=0.01))
        model.add(BatchNormalization(epsilon=1e-05, axis=1, momentum=0.99, weights=None, beta_initializer='zeros', gamma_initializer='ones'))
        model.add(MaxPooling3D(pool_size=(nb_pool, nb_pool, nb_pool)))

#  Next Convolution3D block
        model.add(Convolution3D(nb_filters[1], conv_size2, padding='valid'))
        if activation == 'relu':
            model.add(Activation('relu'))
        if activation == 'leakyrelu':
            model.add(LeakyReLU(alpha=0.01))
        if activation == 'elu':
            model.add(ELU(alpha=0.01))
        model.add(BatchNormalization(epsilon=1e-05, axis=1, momentum=0.99, weights=None, beta_initializer='zeros', gamma_initializer='ones'))
        model.add(Dropout(0.25))
        model.add(MaxPooling3D(pool_size=(nb_pool, nb_pool, nb_pool)))

#  Additional Convolution3D block
        if model_no == 3:
            model.add(Convolution3D(nb_filters[2], conv_size3, padding='valid'))  # EXTRA CONV LAYER
            if activation == 'relu':
                model.add(Activation('relu'))
            if activation == 'leakyrelu':
                model.add(LeakyReLU(alpha=0.01))
            if activation == 'elu':
                model.add(ELU(alpha=0.01))
            model.add(BatchNormalization(epsilon=1e-05, axis=1, momentum=0.99, weights=None, beta_initializer='zeros', gamma_initializer='ones'))
            model.add(Dropout(0.25))
            model.add(MaxPooling3D(pool_size=(nb_pool, nb_pool, nb_pool)))
        model.add(Flatten())
        model.add(Dropout(0.25))
        model.add(Dense(nb_classes, kernel_initializer='normal'))
        model.add(Activation('softmax'))


    print('... compiling model')
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=["accuracy"])
    model.summary()
    return model
