"""
@author: Nicholas Nordstrom
main.py
    Main thread of execution

"""
import glob
import os
import time
import pandas as pd
import numpy as np
import cv2
import tensorflow as tf
from keras import models, layers, losses, callbacks


def exclude_image(paths, verbose=False):
    """
    method to exclude images from the dataset whose paths are passed
    :param verbose: option to print which groups of files are excluded
    :param paths: path of file to exclude
    :return: tuple(number of files excluded, time to exclude)
    """
    start = time.time()
    n = 0
    for path in paths:
        if verbose:
            print("Excluding {} ({} images)".format(path[0].split('\\')[-1][:7], len(path)))
        for p in path:
            n += 1
            os.replace(p.replace('\\', '/'), "data/Excluded/" + p.split('\\')[-1])
    return n, time.time() - start


def get_data(n=None):
    """
    method to import data and get it all nice and ready for learning.
    :param n: number of image sets to obtain
    :return: our dataset.
    """
    start = time.time()
    if n is None:
        paths = [glob.glob("data/FingerJoints/" + x[0:7] + "*") for x in os.listdir("data/FingerJoints/")[0::12]]
    else:
        paths = [glob.glob("data/FingerJoints/" + x[0:7] + "*") for x in os.listdir("data/FingerJoints/")[0:12 * n:12]]
    
    file = pd.read_excel('test.xlsx')
    file = file.set_index('id')
    
    label = []
    for i in range(len(paths)):
        id = paths[i][0][18:25]
        id = int(id)
        for j in range(len(paths[i])):
            joint = paths[i][j][26:30]
            kl = file.at[id, joint]
            kl = int(kl)
            label.append(kl)

    label = np.array(label, dtype=np.uint8)
    return paths, label, time.time() - start


def preprocess(data):
    """
    preprocesses our data so that it is more easily understood by our algorithm.
    :return: preprocessed data.
    """
    start = time.time()
    return data, time.time() - start


def data_augment(data):
    """
    performs data augmentation as needed.
    :param data: data to augment.
    :return: augmented form of the data.
    """
    start = time.time()
    return data, time.time() - start


def cnn_VGG16():
    """
    creates our algorithm to learn from our dataset.
    :return: the model object.
    """
    start = time.time()
    model = models.Sequential()
    model.add(layers.Conv2D(input_shape=(180, 180, 1), filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(layers.Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(layers.Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(layers.Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(layers.Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(layers.Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(layers.Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(layers.Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(layers.Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(layers.Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(units=4096, activation="relu"))
    model.add(layers.Dense(units=4096, activation="relu"))
    model.add(layers.Dense(units=1, activation="softmax"))

    model.compile(optimizer='adam', loss=losses.categorical_crossentropy, metrics=['accuracy'])
    return model, time.time() - start


def pipeline(paths, y):
    """
    This is our data 'pipeline' it ensures a consistent flow and modulation of the data. implements lazy image loading.
    :return:
    """
    start = time.time()
    X = np.array([[cv2.imread(p)[:, :, 0] for p in path] for path in paths])
    X, ttt = preprocess(X)
    X_aug, tta = data_augment(X)
    return X, y, X_aug, ttt, tta, time.time() - start


def cross_validation(model, X, y, X_aug, n=10, verbose=False):
    """
    Performs n-fold cross validation on X, y pairs of data
    :param verbose: Option to print details for validation timings
    :param X_aug: augmented input data in the same order as original data X
    :param model: model to cross validate
    :param n: number of folds for cross validation
    :param X: input data for cross validation
    :param y: output data for cross validation
    :return: accuracy result from cross validation
    """
    start = time.time()
    callback = callbacks.EarlyStopping(monitor='loss', patience=3)

    step = y.shape[0]//n
    av_accuracy = 0
    for fold in range(n):
        # define training set
        training = np.zeros((X.shape[0]), dtype=bool)
        training[fold*step:step*(fold+1)] = 1

        # fit to training data
        model.fit(X[training], y[training], callbacks=[callback])

        # test on remaining data
        _, acc = model.evaluate(X[np.logical_not(training)], y[np.logical_not(training)])
        av_accuracy += (acc/n)

    return av_accuracy, time.time()-start


def main():
    n = 1
    while n:
        # Get chunk of data
        start = time.time()

        paths, y, ttr = get_data(50)
        print("Getting Data Paths Took {} Seconds!".format(round(ttr, 4)))

        # Data exclusion
        missing = [p for p in paths if len(p) != 12]
        if len(missing) > 0:
            print("< < < DATA IMPURITY FOUND: EXCLUDING & RETRIEVING... > > >")
        n, tte = exclude_image(missing, verbose=True)
        print("Excluded {} Files In {} Seconds!".format(n, round(tte, 4)))
    ttc = time.time()-start
    print("Total Data Cleaning Took {} Seconds!".format(round(ttc, 4)))

    # Send data to the pipeline
    X, y, X_aug, ttt, tta, ttp = pipeline(paths, y)
    print("Data Transformation Took {} Seconds!".format(round(ttt, 4)))
    print("Data Augmentation Took {} Seconds!".format(round(tta, 4)))
    print("Total Data Pipeline Took {} Seconds!".format(round(ttp, 4)))

    # Create our model
    model, ttc = cnn_VGG16()
    print("Creating Model Took {} Seconds!".format(round(ttc, 4)))

    # Testing Data Shape
    print("X shape: {}".format(X.shape))
    print("y shape: {}".format(y.shape))

    # check data type
    print("X data type is {}".format(X.dtype))
    print("y data type is {}".format(y.dtype))
    print("X[0] data type is {}".format(X[0].dtype))
    print("y[0] data type is {}".format(y[0].dtype))

    # Train model on training data
    accuracy, ttv = cross_validation(model, X.reshape(y.shape[0], 180, 180), y.reshape(600, 1))
    print("Model Scored {}% Accuracy, In {} Seconds!".format(accuracy, ttv))


if __name__ == "__main__":
    main()
