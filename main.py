"""
@author: Nicholas Nordstrom
main.py
    Main thread of execution

"""
import glob
import os
import time
import scipy
from keras_preprocessing.image import ImageDataGenerator
import pandas as pd
import numpy as np
import cv2
from keras import models, layers, losses, callbacks
from tensorflow.keras.applications import *
from tensorflow.keras import optimizers
import tensorflow as tf
import matplotlib.pyplot as plt


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


def inception_v3(input_shape, output_shape, verbose=False, loss=losses.binary_crossentropy):
    """
    :param loss: loss function to calculate loss between epochs
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
    model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
    if verbose:
        model.summary()
    return model, time.time() - start


def dense_net201(input_shape, output_shape, verbose=False, loss=losses.binary_crossentropy):
    """
    :param loss: loss function to calculate loss between epochs
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
    model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
    if verbose:
        model.summary()
    return model, time.time() - start


def efficient_net(input_shape, output_shape, verbose=False, loss='binary_crossentropy'):
    """
    :param loss: loss function to calculate loss between functions
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
        loss=loss,
        optimizer=optimizers.RMSprop(lr=2e-5),
        metrics=["accuracy"])
    if verbose:
        model.summary()
    return model, time.time() - start


def cnn_vgg16(input_shape, output_shape, verbose=False, loss='binary_crossentropy'):
    """
    :param loss: loss function to calculate loss between epochs
    :@author: https://towardsdatascience.com/step-by-step-vgg16-implementation-in-keras-for-beginners-a833c686ae6c
    creates our algorithm to learn from our dataset.
    :param input_shape: shape of input for model
    :param output_shape: shape of output
    :param verbose: option to print details about model
    :return: the model object.
    """
    start = time.time()
    model = models.Sequential()
    model.add(VGG16(include_top=False, weights=None, input_tensor=None, input_shape=input_shape, pooling=None, classes=output_shape))
    model.add(layers.Flatten())
    model.add(layers.Dense(units=4096, activation="relu"))
    model.add(layers.Dense(units=4096, activation="relu"))
    model.add(layers.Dense(units=1, activation="softmax"))
    model.compile(optimizer='adam', loss=loss, metrics=['accuracy', 'binary_accuracy'])
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
    start = time.time()
    callback = callbacks.EarlyStopping(monitor='loss', patience=3)
    X, y = randomize(X, y, random_state)

    training = np.zeros(X.shape[0], dtype=bool)
    validate = np.zeros(X.shape[0], dtype=bool)
    testing = np.zeros(X.shape[0], dtype=bool)

    training[:int(split[0]*X.shape[0])] = 1
    validate[int(split[0]*X.shape[0]):int((split[0]+split[1])*X.shape[0])] = 1
    testing[int((split[0]+split[1])*X.shape[0]):] = 1

    train_score = model.fit(X[training], y[training], epochs=5)
    test_score = model.evaluate(X[testing], y[testing])
    # val_hist, val_score = model.evaluate(X[validate], y[validate])
    # val_score = model.fit(X[validate], y[validate], epochs=50)
    # test_hist, test_score = model.evaluate(X[testing], y[testing])
    print(model.predict(X[0].reshape(1, 180, 180, 1)))

    return train_score.history  # train_hist, val_hist, test_hist, val_score, test_score, time.time()-start


def train_test(model, df, split=.8, random_state=None):
    """
    tests a model with a random permutation of the data
    :param model: model to test
    :param random_state: random state for data permutation
    :return: score of the model
    """
    start = time.time()

    callback = callbacks.EarlyStopping(monitor='loss', patience=3)

    oa0 = df[df['oa'] == 0].head(3500)
    oa1 = df[df['oa'] == 1].head(3500)

    df_train = pd.concat([oa0.iloc[:int(split * len(oa0))], oa1.iloc[:int(split * len(oa1))]])
    df_test = pd.concat([oa0.iloc[int(split * len(oa0)):], oa1.iloc[int(split * len(oa1)):]])

    print("Input y shape: {}".format(df_train["oa"].shape))
    print("Input y: {}".format(df_train["oa"]))

    train_generator = ImageDataGenerator().flow_from_dataframe(
        dataframe=df_train,
        x_col="path",
        y_col="oa",
        class_mode="raw",
        shuffle=True,
        target_size=(180, 180),
        color_mode='grayscale')
    test_generator = ImageDataGenerator().flow_from_dataframe(
        dataframe=df_test,
        x_col="path",
        y_col='oa',
        shuffle=True,
        class_mode='raw',
        target_size=(180, 180),
        color_mode='grayscale')

    hist_train = model.fit(train_generator, epochs=10)
    hist_test = model.evaluate(test_generator)

    test_generator.reset()

    pred = model.predict(test_generator)
    actual = df_test['oa'].to_numpy()

    print("pred shape: {}".format(pred.shape))
    print("actual shape: {}".format(actual.shape))

    print("pred: {}".format(pred))
    print("actual: {}".format(actual))

    return hist_train.history, hist_test, tf.math.confusion_matrix(actual, pred), time.time()-start


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


def generate_data(X, y, batch_size=32):
    cur_batch = 0
    image_batch = []
    label_batch = []
    for b in range(batch_size):
        image_batch.append(cv2.imread(X[cur_batch*batch_size+b]))
        label_batch.append(y[cur_batch*batch_size+b])
    cur_batch += 1
    yield np.array(image_batch), np.array(label_batch)


def main():
    verbose = 1

    # Get DataFrame
    df, ttr = get_data()
    print("Reading Excel File Took {} Seconds!".format(round(ttr, 4)))

    # Understand the data
    if verbose:
        print("< < < Data Analytics > > >")
        print("There Are {} Total Data Points".format(len(df)))
        print("There Are {} OA Joints And {} non-OA joints".format(len(df[df['oa'] == 1]), len(df[df['oa'] == 0])))
        print("There Are {} MCP, {} PIP, And {} DIP Joints".format(len(df[df['joint'] == 'mcp']), len(df[df['joint'] == 'pip']), len(df[df['joint'] == 'dip'])))
        print("There Are {} KL0, {} KL1, {} KL2, {} KL3, And {} KL4".format(len(df[df['kl'] == 0]), len(df[df['kl'] == 1]), len(df[df['kl'] == 2]), len(df[df['kl'] == 3]), len(df[df['kl'] == 4])))

    # Create our model
    model, ttc = cnn_vgg16((180, 180, 1), 1, verbose=True)
    print("Creating Model Took {} Seconds!".format(round(ttc, 4)))

    # kl0 = df[df['kl'] == 0].iloc[:100]
    # kl1 = df[df['kl'] == 1].iloc[:100]
    # kl2 = df[df['kl'] == 2].iloc[:100]
    # kl3 = df[df['kl'] == 3].iloc[:100]
    # kl4 = df[df['kl'] == 4].iloc[:100]
    # df_train = pd.concat([kl0.iloc[:80], kl1.iloc[:80], kl2.iloc[:80], kl3.iloc[:80], kl4.iloc[:80]])
    # df_test = pd.concat([kl0.iloc[80:], kl1.iloc[80:], kl2.iloc[80:], kl3.iloc[80:], kl4.iloc[80:]])

    # Test model
    hist_train, hist_test, cm, ttt = train_test(model, df, .8)
    print(cm)
    print("Training History: {}".format(hist_train))
    print("Testing History: {}".format(hist_test))
    print("Model Training Took {} Seconds!".format(round(ttt, 4)))

    # Plot Results
    # plt.plot()

    # train model on training data
    # train_acc = train_test_validate(model, X, y, [.8, .1, .1], random_state=42)
    # print("Model Scored {}% Training Accuracy, In {} Seconds!".format(train_acc,  round(0, 4)))
    # print("Model Scored {}% Validation Accuracy, In {} Seconds!".format(val_score, round(ttm, 4)))
    # print("Model Scored {}% Testing Accuracy, In {} Seconds!".format(test_score, round(ttm, 4)))


if __name__ == "__main__":
    main()
