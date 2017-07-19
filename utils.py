import matplotlib
import matplotlib.pyplot as plt
import joblib
import csv
import numpy as np
import os
import multiprocessing
from keras.utils import np_utils, generic_utils

from joblib import Parallel, delayed
num_cores = multiprocessing.cpu_count() - 5



################################
# Function for saving a figure #
################################
def savefigure(path, ext='png', close=True, verbose=True):
    """Save a figure from pyplot.
    Parameters
    ----------
    path : string
        The path (and filename, without the extension) to save the
        figure to.
    ext : string (default='png')
        The file extension. This must be supported by the active
        matplotlib backend (see matplotlib.backends module).  Most
        backends support 'png', 'pdf', 'ps', 'eps', and 'svg'.
    close : boolean (default=True)
        Whether to close the figure after saving.  If you want to save
        the figure multiple times (e.g., to multiple formats), you
        should NOT close it in between saves or you will have to
        re-plot it.
    verbose : boolean (default=True)
        Whether to print information about when and where the image
        has been saved.
    """

    # Extract the directory and filename from the given path
    directory = os.path.split(path)[0]
    filename = "%s.%s" % (os.path.split(path)[1], ext)
    if directory == '':
        directory = '.'

    # If the directory does not exist, create it
    if not os.path.exists(directory):
        os.makedirs(directory)

    # The final path to save to
    savepath = os.path.join(directory, filename)

    if verbose:
        print("Saving figure to '%s'..." % savepath),

    # Actually save the figure
    plt.savefig(savepath)

    # Close it
    if close:
        plt.close()

    if verbose:
        print("Done")

def proccounter(x, y, i, c, num,y_total,data_path,voxel_count,p):
    t_str = "%d" % num[i] + '.raw'
    u = os.path.join(data_path, p[c])
    filename = os.path.join(u, t_str)
    temp_array = np.fromfile(filename, dtype=np.dtype('uint8'))
    temp_array= np. reshape((voxel_count, voxel_count, voxel_count),order='C')
    temp_array = temp_array.astype('float') / 255.0
    x[i, :, :, :, c] = temp_array
    y[i] = y_total[num[i]]
    del temp_array

def load_test_data(channels, voxel_count,mode, nb_classes):
    code_path = os.path.dirname(os.path.realpath(__file__))
    project_path, code_folder = os.path.split(code_path)
    if (mode=="training"):
        data_path = os.path.join(project_path, 'Data/training')
    else:
        data_path = os.path.join(project_path, 'Data/testing')

    # data params
    f = open((os.path.join(data_path, "outputs.csv")), 'r')
    reader1 = csv.reader(f,delimiter=',')
    data_size=sum(1 for line in reader1)
    data_size-=1
    f.close()
    print(data_size)

    outputs=-np.ones(data_size)

    f = open((os.path.join(data_path, "outputs.csv")), 'r')
    reader = csv.reader(f,delimiter=',')
    for i, row in enumerate(reader):
        if (i>0):
            outputs[i-1] = row[13]
    f.close()
    print('outputs loaded')

    num = list(range(data_size))
    np.random.shuffle(num)

    if (mode=="training"):
        validation_split = 0.2
        training_split = 1 - validation_split

    if (channels == 1):
        p = ['inouts']
    elif (channels == 3):
        p = ['xNormalsio', 'yNormalsio', 'zNormalsio']
    elif (channels == 4):
        p = ['inouts', 'xNormals', 'yNormals', 'zNormals']


    if (mode=="training"):
        train_num = num[:int(training_split * data_size)]
        validation_num = num[int(training_split * data_size):data_size]

        # load validation data
        y_val = -np.ones(int(validation_split * data_size))
        x_val = -np.ones([int(validation_split * data_size), voxel_count, voxel_count, voxel_count, channels])
        Parallel(n_jobs=num_cores, backend='threading', verbose=1)(delayed(proccounter)(x_val, y_val, counter, channel, validation_num,outputs,data_path,voxel_count,p) for counter in range(int(validation_split * data_size)) for channel in range(channels))

        x_train = -np.ones([(int(training_split * data_size)), voxel_count, voxel_count, voxel_count, channels])
        y_train = -np.ones([(int(training_split * data_size))])
        Parallel(n_jobs=num_cores, backend='threading', verbose=1)(delayed(proccounter)(x_train, y_train, counter, channel, train_num,outputs,data_path,voxel_count,p) for counter in range(int(training_split * data_size)) for channel in range(channels))

        y_train = np_utils.to_categorical(y_train, nb_classes)
        y_val = np_utils.to_categorical(y_val, nb_classes)

        return x_train, y_train, x_val, y_val, train_num, validation_num

    if (mode=="testing"):
        # load test data
        e = 0
        y_test = -np.ones((data_size))
        x_test = -np.ones([data_size, voxel_count, voxel_count, voxel_count, channels])
        Parallel(n_jobs=num_cores, backend='threading', verbose=1)(delayed(proccounter)(x_test, y_test, counter, channel, num,outputs,data_path,voxel_count,p) for counter in range(int(validation_split * data_size)) for channel in range(channels))
        return x_test,y_test,num
