import tensorflow as tf
import numpy as np
from keras import backend as K
from keras.models import Sequential
from keras.layers import Lambda
def loss_calculation(x, category_index, nb_classes):
    return tf.multiply(x, K.one_hot((category_index), nb_classes))

def loss_calculation_shape(input_shape):
    return input_shape

def normalize(x):
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

def prepareGradCAM(input_model,input_layer_index,nb_classes):
    model = input_model
    #  because non-manufacturability is 1
    explanation_catagory = 1
    loss_function = lambda x: loss_calculation(x, 1, nb_classes)
    model.add(Lambda(loss_function,
                     output_shape = loss_calculation_shape))
    #  use the loss from the layer before softmax. As best practices
    loss = K.sum(model.layers[-1].output)
    # last fully Convolutional layer to use for computing GradCAM
    conv_output = model.layers[-5].output
    grads = normalize(K.gradients(loss, conv_output)[0])
    gradient_function = K.function([model.layers[0].input, K.learning_phase()], [conv_output, grads])

    return gradient_function


def GradCAM(gradient_function, input_file):
    explanation_catagory=1
    # Shape of the fully convolutional layer to use
    f=5
    output, grads_val = gradient_function([input_file, 0])
    output = output[0, :]
    # print(grads_val.shape)
    weights = np.mean(grads_val, axis=(1, 2, 3))
    # print('weights', weights)
    grad_cam = np.ones(output.shape[0:3], dtype=np.float32)

    for i, w in enumerate(np.transpose(weights)):
        grad_cam += w * output[:, :, :, i]

    grad_cam = np.maximum(grad_cam, 0)
    print(weights)
    grad_cam = grad_cam / np.max(grad_cam)
    attMap = np.zeros(input_file.shape[1] *
                    input_file.shape[2] * input_file.shape[3])
    grad_cam = np.reshape(grad_cam, (f * f * f),order='C')


#  Linearly interpolating the grad_cam to the size of input_file to get attMap
    input_file = np.reshape(input_file, (64 * 64 * 64),order='C')
    attMap = np.reshape(attMap, (64 * 64 * 64),order='C')
    for k in range(64):
        for j in range(64):
            for i in range(64):
                i1 = int((i / 64.0) * f)
                j1 = int((j / 64.0) * f)
                k1 = int((k / 64.0) * f)
                attMap[64 * 64 * k + 64 * j + i] = grad_cam[f * f * k1 + f * j1 + i1]
                if (input_file[64 * 64 * k + 64 * j + i] == 0):
                    attMap[64 * 64 * k + 64 * j + i] = 0

    attMap = (1 * np.float32(attMap)) + (1 * np.float32(input_file))

    attMap = (attMap / np.max(attMap))
    return np.reshape(attMap,(64,64,64),order='C')
