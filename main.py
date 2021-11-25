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
from keras import models, layers, losses, callbacks
from tensorflow.keras.applications import *
from tensorflow.keras import optimizers


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


def get_data():
    """
    method to import data and get it all nice and ready for learning.
    :return: pandas DataFrame of the dataset
    """
    start = time.time()
    df = pd.read_excel('data/data.xlsx')
    return df, time.time() - start


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
    return [], time.time() - start


def inception_v3(input_shape, output_shape, verbose=False):
    """
    :@author: https://docs.w3cub.com/tensorflow~python/tf/keras/applications/inceptionV3
    Creates an InceptionV3 model
    :param input_shape: shape of input layer
    :param output_shape: shape of output layer
    :param verbose: option to print model summary to console
    :return: compiled and ready-to-train model
    """

    start = time.time()
    model = InceptionV3(include_top=True,
                                                        weights=None,
                                                        input_tensor=None,
                                                        input_shape=input_shape,
                                                        pooling=None,
                                                        classes=output_shape)
    model.compile(optimizer='adam', loss=losses.sparse_categorical_crossentropy, metrics=['accuracy'])
    if verbose:
        model.summary()
    return model, time.time() - start


def dense_net201(input_shape, output_shape, verbose=False):
    """
    :@author: https://docs.w3cub.com/tensorflow~python/tf/keras/applications/densenet201
    Creates a DenseNet201 model
    :param input_shape: shape of input layer
    :param output_shape: shape of output layer
    :param verbose: option to print model summary to console
    :return: compiled and ready-to-train model
    """
    start = time.time()
    model = DenseNet201(include_top=True,
                                                    weights=None,
                                                    input_tensor=None,
                                                    input_shape=input_shape,
                                                    pooling=None,
                                                    classes=output_shape)
    model.compile(optimizer='adam', loss=losses.sparse_categorical_crossentropy, metrics=['accuracy'])
    if verbose:
        model.summary()
    return model, time.time() - start


def efficient_net(input_shape, output_shape, verbose=False):
    """
    :@author: https://towardsdatascience.com/an-in-depth-efficientnet-tutorial-using-tensorflow-how-to-use-efficientnet-on-a-custom-dataset-1cab0997f65c
    Creates a efficientNet model, loads trained weights as a starting point
    :param input_shape: shape of input
    :param output_shape: shape of output
    :param verbose: option to print model summary
    :return: compiled model
    """
    start = time.time()
    conv_base = EfficientNetB6(include_top=False,
                                                        weights=None,
                                                        input_tensor=None,
                                                        input_shape=input_shape,
                                                        pooling=None,
                                                        classes=output_shape)
    model = models.Sequential()
    model.add(conv_base)
    model.add(layers.GlobalMaxPooling2D(name="gap"))
    model.add(layers.Dropout(rate=0.2, name="dropout_out"))
    model.add(layers.Dense(output_shape, activation="softmax", name="fc_out"))
    conv_base.trainable = False
    model.compile(
        loss=losses.sparse_categorical_crossentropy,
        optimizer=optimizers.RMSprop(lr=2e-5),
        metrics=["accuracy"])
    if verbose:
        model.summary()
    return model, time.time() - start


def cnn_vgg16(input_shape, output_shape, verbose=False):
    """
    :@author: https://towardsdatascience.com/step-by-step-vgg16-implementation-in-keras-for-beginners-a833c686ae6c
    creates our algorithm to learn from our dataset.
    :param input_shape: shape of input for model
    :param output_shape: shape of output
    :param verbose: option to print details about model
    :return: the model object.
    """
    start = time.time()
    model = models.Sequential()
    model.add(layers.Conv2D(input_shape=input_shape, filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
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
    model.add(layers.Dense(units=output_shape, activation="softmax"))

    model.compile(optimizer='adam', loss=losses.sparse_categorical_crossentropy, metrics=['accuracy'])
    if verbose:
        model.summary()
    return model, time.time() - start


def pipeline(paths, y):
    """
    This is our data 'pipeline' it ensures a consistent flow and modulation of the data. implements lazy image loading.
    :param paths: path data for image input
    :param y: truth label for data
    :return:tuple of original X, y, augmented X, time to transform, time to augment, total time to pipeline
    """
    start = time.time()
    X = np.array([cv2.imread(p)[:, :, 0] for p in paths])
    X, ttt = preprocess(X)
    X_aug, tta = data_augment(X)
    return X, y, X_aug, ttt, tta, time.time() - start


def randomize(X, y, random_state=None):
    """
    randomize order of items in unison
    :param X: array1
    :param y: array2
    :param random_state: random state for permutations
    :return: random permutations of X, y pairs
    """
    if random_state is None:
        shuffle = np.random.RandomState().permutation(y.shape[0])
    else:
        shuffle = np.random.RandomState(seed=random_state).permutation(y.shape[0])

    return np.array([X[i] for i in shuffle]), np.array([y[i] for i in shuffle])


def train_test_validate(model, X, y, split=[.8, .1, .1], random_state=None):
    X, y = randomize(X, y, random_state)
    model.fit(X[0:split*.8], y[0:split[0]])


def test(model, X, y, random_state=None):
    """
    tests a model with a random permutation of the data
    :param model: model to test
    :param X: input data
    :param y: output labels
    :param random_state: random state for data permutation
    :return: score of the model
    """
    X, y = randomize(X, y, random_state=random_state)
    return model.evaluate(X, y)[1]


def cross_validation(model, X, y, X_aug, n=10, verbose=False, random_state=None):
    """
    Performs n-fold cross validation on X, y pairs of data
    :param random_state: random state for randomization
    :param verbose: Option to print details for validation timings
    :param X_aug: augmented input data in the same order as original data X
    :param model: model to cross validate
    :param n: number of folds for cross validation
    :param X: input data for cross validation
    :param y: output data for cross validation
    :return: accuracy result from cross validation
    """
    start = time.time()
    model.save('fresh_model')

    X, y = randomize(X, y, random_state=random_state)

    callback = callbacks.EarlyStopping(monitor='loss', patience=3)
    step = y.shape[0]//n
    av_accuracy = 0
    for fold in range(n):
        # load new model starting from scratch
        model = models.load_model('fresh_model')

        # define training set
        training = np.ones((X.shape[0]), dtype=bool)
        training[fold*step:step*(fold+1)] = 0

        # fit to training data
        start_train = time.time()
        model.fit(X[training], y[training], epochs=50, callbacks=[callback])
        tt_train = time.time() - start_train
        if verbose:
            print("Fold {} Finished Training On {} Data-points In {} Seconds!".format(fold, sum(training),
                                                                                      round(tt_train, 2)))

        for aug in X_aug:
            start_train = time.time()
            model.fit(aug[training], y[training], epochs=50, callbacks=[callback])
            tt_train = time.time() - start_train
            if verbose:
                print("Fold {} Finished Training On {} Augmented Data-points In {} Seconds!".format(fold, sum(training),
                                                                                          round(tt_train, 2)))

        # test on remaining data
        start_test = time.time()
        _, acc = model.evaluate(X[np.logical_not(training)], y[np.logical_not(training)])
        acc *= 100
        av_accuracy += (acc/n)
        tt_test = time.time() - start_test
        if verbose:
            print("Fold {} Finished Testing On {} Data-points In {} Seconds, With {}% Accuracy".format(fold, sum(np.logical_not(training)), round(tt_test, 4), round(acc, 2)))

    return round(av_accuracy, 2), time.time()-start


def main():
    verbose = 1

    # Get DataFrame
    df, ttr = get_data()
    print("Reading Excel File Took {} Seconds!".format(round(ttr, 4)))

    # Understand the data
    if verbose:
        print("< < < Data Analytics > > >")
        print("There Are {} OA Joints And {} non-OA joints".format(len(df[df['oa'] == 1]), len(df[df['oa'] == 0])))
        print("There Are {} MCP, {} PIP, And {} DIP Joints".format(len(df[df['joint'] == 'mcp']), len(df[df['joint'] == 'pip']), len(df[df['joint'] == 'dip'])))
        print("There Are {} KL0, {} KL1, {} KL2, {} KL3, And {} KL4".format(len(df[df['kl'] == 0]), len(df[df['kl'] == 1]), len(df[df['kl'] == 2]), len(df[df['kl'] == 3]), len(df[df['kl'] == 4])))


    # Send data to the pipeline
    X, y, X_aug, ttt, tta, ttp = pipeline(df.head(100)['path'], df.head(100)['oa'])
    print("Data Transformation Took {} Seconds!".format(round(ttt, 4)))
    print("Data Augmentation Took {} Seconds!".format(round(tta, 4)))
    print("Total Data Pipeline Took {} Seconds!".format(round(ttp, 4)))

    # Create our model
    model, ttc = cnn_vgg16((180, 180, 1), 1, verbose=True)
    print("Creating Model Took {} Seconds!".format(round(ttc, 4)))

    # Train model on training data
    accuracy, ttv = cross_validation(model, X, y, X_aug, verbose=True)
    print("Model Scored {}% Accuracy, In {} Seconds!".format(accuracy, ttv))


if __name__ == "__main__":
    main()
