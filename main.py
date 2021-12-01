"""
@author: Nicholas Nordstrom
main.py
    Main thread of execution

"""
import glob
import os
import time
import keras.applications.vgg16
import keras_preprocessing.image
from scipy.ndimage import zoom
from keras_preprocessing.image import ImageDataGenerator
import pandas as pd
import numpy as np
import cv2
from keras import models, layers, losses, callbacks
from tensorflow.keras.applications import vgg16, inception_v3, efficientnet, densenet
from tensorflow.keras import optimizers
import tensorflow as tf
import matplotlib.pyplot as plt


global zoom_scale


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
    model = inception_v3.InceptionV3(include_top=True,
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
    model = densenet.DenseNet201(include_top=True,
                                                    weights=None,
                                                    input_tensor=None,
                                                    input_shape=input_shape,
                                                    pooling=None,
                                                    classes=output_shape)
    model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
    if verbose:
        model.summary()
    return model, time.time() - start


def preprocess_zoom(img, scale=None):
    if scale is None:
        global zoom_scale
        scale = zoom_scale
    # resize image
    h, w = img.shape
    img = cv2.resize(img, (h * scale, w * scale), interpolation=cv2.INTER_AREA)

    # crop image
    x = img.shape[1]//2 - w // 2
    y = img.shape[0]//2 - h // 2

    return img[int(y):int(y + h), int(x):int(x + w)]


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
    conv_base = efficientnet.EfficientNetB6(include_top=False,
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
        optimizer=optimizers.RMSprop(learning_rate=2e-5),
        metrics=["accuracy"])
    if verbose:
        model.summary()
    return model, time.time() - start


def cnn_vgg16(input_shape, output_shape, verbose=False, loss='binary_crossentropy', activation='softmax', optimizer='adam'):
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
    model = vgg16.VGG16(
        weights=None,
        input_tensor=None,
        input_shape=input_shape,
        pooling=None,
        classes=output_shape,
        classifier_activation=activation)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy', 'binary_accuracy'])
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


def generate_data_binary(df, train=.8, test=.2, max_data=3500):
    """
    create data generators with train-test-validate split using max_data rows from DataFrame based on binary oa label
    :param df: DataFrame with data to flow from
    :param train: percentage (out of one) for training set
    :param test: percentage (out of one) for testing set
    :param max_data: max rows of data to use
    :return: tuple of training data generator, testing data generator and time taken
    """
    start = time.time()

    oa0 = df[df['oa'] == 0].head(max_data)
    oa1 = df[df['oa'] == 1].head(max_data)

    df_train = pd.concat([
        oa0.iloc[:int(train * len(oa0))],
        oa1.iloc[:int(train * len(oa1))]])

    # df_val = pd.concat([
    #     oa0.iloc[int((train+test) * len(oa0)):],
    #     oa1.iloc[int((train+test) * len(oa0)):]])

    df_test = pd.concat([
        oa0.iloc[int(train * len(oa0)):int((train+test) * len(oa0))],
        oa1.iloc[int(train * len(oa1)):int((train+test) * len(oa0))]])

    generators = [ImageDataGenerator().flow_from_dataframe(
        dataframe=x,
        x_col="path",
        y_col="oa",
        class_mode="raw",
        shuffle=True,
        target_size=(180, 180),
        preprocess=None,
        color_mode='grayscale') for x in [df_train, df_test]]

    return generators[0], generators[1], df_test['oa'], time.time() - start


def generate_data_multiclass(df, train=.8, test=.2, max_data=100):
    """
    create data generators with train-test-validate split using max_data rows from DataFrame based on multiclass kl label
    :param df: DataFrame with data to flow from
    :param train: percentage (out of one) for training set
    :param test: percentage (out of one) for testing set
    :param max_data: max rows of data to use
    :return: tuple of training data generator, testing data generator and time taken
    """
    start = time.time()

    kl0 = df[df['kl'] == 0].head(max_data)
    kl1 = df[df['kl'] == 1].head(max_data)
    kl2 = df[df['kl'] == 2].head(max_data)
    kl3 = df[df['kl'] == 3].head(max_data)
    kl4 = df[df['kl'] == 4].head(max_data)

    df_train = pd.concat([
        kl0.iloc[:int(train * len(kl0))],
        kl1.iloc[:int(train * len(kl1))],
        kl1.iloc[:int(train * len(kl2))],
        kl1.iloc[:int(train * len(kl3))],
        kl1.iloc[:int(train * len(kl4))]])

    # df_val = pd.concat([
    #     kl0.iloc[int((train + test) * len(kl0)):],
    #     kl1.iloc[int((train + test) * len(kl1)):],
    #     kl2.iloc[int((train + test) * len(kl2)):],
    #     kl3.iloc[int((train + test) * len(kl3)):],
    #     kl4.iloc[int((train + test) * len(kl4)):]])

    df_test = pd.concat([
        kl0.iloc[int(train * len(kl0)):int((train + test) * len(kl0))],
        kl1.iloc[int(train * len(kl1)):int((train + test) * len(kl1))],
        kl2.iloc[int(train * len(kl2)):int((train + test) * len(kl2))],
        kl3.iloc[int(train * len(kl3)):int((train + test) * len(kl3))],
        kl4.iloc[int(train * len(kl4)):int((train + test) * len(kl4))]])

    generators = [ImageDataGenerator().flow_from_dataframe(
            dataframe=x,
            x_col="path",
            y_col="kl",
            class_mode="raw",
            shuffle=True,
            preprocess_function=None,
            target_size=(180, 180),
            color_mode='grayscale') for x in [df_train, df_test]]

    return generators[0], generators[1], df_test['kl'], time.time() - start


def generate_data(df, train, test, classification_type):
    """
    Creates data generators for either binary or multiclass classification
    :param df: DataFrame source of data
    :param train: percentage (out of one) for training set
    :param test: percentage (out of one) for testing set
    :param classification_type: either 'binary' or 'multiclass'
    :return: data generators and timings
    """
    if classification_type == 'binary':
        return generate_data_binary(df, train, test)
    return generate_data_multiclass(df, train, test)


def train_test(model, gen_train, gen_test, epochs=None):
    """
    trains and tests model
    :param model: model to work with
    :param gen_train: training data generata
    :param gen_val: validation data generator
    :param gen_test: testing data generata
    :return: tuple of history metrics, confusion matrix, and timings
    """
    start = time.time()

    if epochs is None:
        hist_train = model.fit(gen_train, callbacks=[callbacks.EarlyStopping(monitor='loss', patience=3)])
    else:
        hist_train = model.fit(gen_train, epochs=epochs)
    hist_test = model.evaluate(gen_test)

    gen_test.reset()

    pred = model.predict(gen_test)

    return hist_train.history, hist_test, pred, time.time()-start


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


def plot_results(history, metric, label2='epoch', file_name=None):
    plt.plot(range(len(history[metric])), history[metric])
    plt.ylabel(metric)
    plt.xlabel(label2)
    if file_name is None:
        plt.show()
    else:
        plt.savefig(file_name)
        plt.close()


def main():
    global zoom_scale
    verbose = 0

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

        plt.figure()

        # subplot(r,c) provide the no. of rows and columns
        f, axarr = plt.subplots(10, 1)

        # use the created array to output your multiple images. In this case I have stacked 4 images vertically
        axarr[0].imshow(preprocess_zoom(cv2.imread(df['path'][0])[:, :, 0], 1))
        axarr[1].imshow(preprocess_zoom(cv2.imread(df['path'][0])[:, :, 0], 2))
        axarr[2].imshow(preprocess_zoom(cv2.imread(df['path'][0])[:, :, 0], 3))
        axarr[3].imshow(preprocess_zoom(cv2.imread(df['path'][0])[:, :, 0], 4))
        axarr[4].imshow(preprocess_zoom(cv2.imread(df['path'][0])[:, :, 0], 5))
        axarr[5].imshow(preprocess_zoom(cv2.imread(df['path'][0])[:, :, 0], 6))
        axarr[6].imshow(preprocess_zoom(cv2.imread(df['path'][0])[:, :, 0], 7))
        axarr[7].imshow(preprocess_zoom(cv2.imread(df['path'][0])[:, :, 0], 8))
        axarr[8].imshow(preprocess_zoom(cv2.imread(df['path'][0])[:, :, 0], 9))
        axarr[9].imshow(preprocess_zoom(cv2.imread(df['path'][0])[:, :, 0], 10))
        plt.show()

    acc = np.zeros((10))

    for k in range(1, 10):
        zoom_scale = k
        # Create our model
        model, ttc = cnn_vgg16(
            (180, 180, 1),
            1,
            loss='binary_crossentropy',
            verbose=True,
            activation='sigmoid',
            optimizer=optimizers.Adam(learning_rate=1e-5))
        print("Creating Model Took {} Seconds!".format(round(ttc, 4)))

        # Get Data Generators
        train, test, truth, ttg = generate_data(df, .8, .2, 'binary')
        print("Data Generator Creation Took {} Seconds!".format(round(ttg, 4)))

        # Test model
        hist_train, hist_test, pred, ttt = train_test(model, train, test, epochs=50)
        print(tf.math.confusion_matrix(truth, pred))
        print("Model Training Took {} Seconds!".format(round(ttt, 4)))

        acc[k] = hist_test[1]
        print("zoom {} had accuracy {}%!".format(k, acc[k]))
        # Plot Results
        plot_results(hist_train, 'loss', file_name='Results/zoom=' + str(k) + "_VGG16_LOSS.png")
        plot_results(hist_train, 'accuracy', file_name='Results/zoom=' + str(k) + "_VGG16_ACCURACY.png")
    print(acc)
    plot_results({'accuracy': acc}, 'accuracy', label2='zoom scale', file_name='Results/zoom_scale_1-10_accuracy.png')


if __name__ == "__main__":
    main()
