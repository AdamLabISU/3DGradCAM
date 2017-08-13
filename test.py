###### ARGUMENT LIST PARSER ######
import os
import argparse
import glob
parser = argparse.ArgumentParser(description='run GRADCAM with a particular model')
parser.add_argument('-m', '--model', type=int, default=1,
                    help='Model 1, 2 or 3')
parser.add_argument('-c','--channels',type=int,default=1,help="Select the representation in terms of channels, 1 or 3 or 4")
parser.add_argument('-a', '--activation', type=str, default='relu', help='Select from relu/leakyrelu/prelu/elu')
parser.add_argument('-e', '--example_no', type=str, default='all', help='Select the model from  the examples folder for evaluation eg. 1,2,3 etc. , else give all')


args = parser.parse_args()
model_no = args.model
activation = args.activation
channels = args.channels
voxelCount = 64
nbClasses = 2
example_no= args.example_no

###### TENSORFLOW INITIALIZATION WITH SINGLE GPU ######
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
from keras import backend as K
K.set_session(sess)

import keras
from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution3D, MaxPooling3D, ZeroPadding3D
from keras.layers.normalization import BatchNormalization

import sys
import numpy as np
import scipy.io as sio
from GradCAM import prepareGradCAM, GradCAM, registerGradient, modifyBackprop, compileSaliencyFunction
from loadModel import loadModel

p=['inouts','xNormals','yNormals','zNormals']

def loadFile(array,filename, xVoxelCount, yVoxelCount, zVoxelCount, channel):
    u = os.path.join(p[c],filename)
    tempArray = np.fromfile(filename, dtype=np.dtype('uint8'))
    tempArray= np. reshape(tempArray,(xVoxelCount, yVoxelCount, zVoxelCount),order='C')
    tempArray = tempArray.astype('float') / 255.0
    array[0, :, :, :, c] = tempArray
    del tempArray



if __name__ == '__main__':

    cnnModel = loadModel(model_no,channels,activation,voxelCount,nbClasses)
    cnnModel.load_weights('weights\model%s_%schannel_%sactivation_%svoxel_count_%sclasses.h5'%(model_no,channels,activation,voxelCount,nbClasses))

    #   Popping the softmax layer as it creates ambiguity in the explanation
    cnnModel.pop()

    #   The layer index of the last fully convolutional layer after removing the softmax layer
    layerIdx = -4

    #   Keras functions for getting the GradCAM and guided GradCAM
    activationFunction=prepareGradCAM(cnnModel, layerIdx, nbClasses)
    saliency_fn = compileSaliencyFunction(cnnModel, model_no, channels, activation, voxelCount, nbClasses, activation_layer=-4)

    if example_no=="all":
        list_raw = glob.glob("Examples\inouts\*.raw")
        for fileidx, filename in enumerate (list_raw):
            array=np.zeros((1,voxelCount,voxelCount,voxelCount,channels))
            for c in range(channels):
                loadFile(array,filename,voxelCount, voxelCount, voxelCount, c)
            predicted_class = cnnModel.predict(array)
            print(filename,'has a predicted class',predicted_class)
            attMap = GradCAM(activationFunction, array)
            gBackprop = saliency_fn([array, 0])
            gGradCam = gBackprop[0] * attMap[..., np.newaxis]
            gGradCam = (gGradCam / np.max(gGradCam))
            finalOutput = (1 * np.float32(gGradCam)) + 1*np.float32(array)
            finalOutput = (finalOutput / np.max(finalOutput))
            finalOutput*=255.0
            finalOutput.astype('uint8').tofile("GradCAM_outputs\\"+filename)
    else:
        filename="Examples/inouts/"+example_no+".raw"
        array=np.zeros((1,voxelCount,voxelCount,voxelCount,channels))
        for c in range(channels):
            loadFile(array,filename,voxelCount, voxelCount, voxelCount, c)
        predicted_class = cnnModel.predict(array)
        print(filename,'has a predicted class',predicted_class)
        attMap = GradCAM(activationFunction, array)
        gBackprop = saliency_fn([array, 0])
        gGradCam = gBackprop[0] * attMap[..., np.newaxis]
        # attMap*=255.0
        gGradCam = (gGradCam / np.max(gGradCam))
        finalOutput = (1 * np.float32(gGradCam)) + (1 * np.float32(array))
        finalOutput = (finalOutput / np.max(finalOutput))
        finalOutput*=255.0
        finalOutput.astype('uint8').tofile("GradCAM_outputs\\"+filename)
        print('attention map saved' )
