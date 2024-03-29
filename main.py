"""
@author: Nicholas Nordstrom, Thurein Myint
main.py
    Main thread of execution

"""
import glob
import os
import time
import keras.applications.vgg16
import keras_preprocessing.image
from sklearn.metrics import roc_curve, auc
from keras_preprocessing.image import ImageDataGenerator
import pandas as pd
import numpy as np
import cv2
from keras import models, layers, losses, callbacks, metrics
from tensorflow.keras.applications import vgg16, inception_v3, efficientnet, densenet, mobilenet
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


def mobile_net(input_shape, output_shape, verbose=False, loss='binary_crossentropy', activation='softmax',
                  optimizer='adam', metrics=['accuracy']):
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
        model = mobilenet.MobileNet(
            weights=None,
            input_tensor=None,
            input_shape=input_shape,
            pooling=None,
            classes=output_shape,
            classifier_activation=activation)
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        if verbose:
            model.summary()
        return model, time.time() - start


def inception_v3(input_shape, output_shape, verbose=False, loss='binary_crossentropy', activation='softmax', optimizer='adam', metrics=['accuracy']):
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
    model = inception_v3.InceptionV3(
        weights=None,
        input_tensor=None,
        input_shape=input_shape,
        pooling=None,
        classes=output_shape,
        classifier_activation=activation)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    if verbose:
        model.summary()
    return model, time.time() - start


def dense_net201(input_shape, output_shape, verbose=False, loss='binary_crossentropy', activation='softmax', optimizer='adam', metrics=['accuracy']):
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
    model = models.sequential.Sequential()
    model.add(densenet.DenseNet201(
        include_top=False,
        weights=None,
        input_tensor=None,
        input_shape=input_shape,
        pooling=None,
        classes=output_shape))

    model.add(layers.Flatten())
    model.add(layers.Dense(output_shape, activation=activation))

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    if verbose:
        model.summary()
    return model, time.time() - start


def preprocess_zoom(img, scale=3):
    # global zoom
    #
    # if zoom is not None:
    #     scale = zoom
    # resize image
    h, w = img.shape
    img = cv2.resize(img, (h * scale, w * scale), interpolation=cv2.INTER_AREA)

    # crop image
    x = img.shape[1]//2 - w // 2
    y = img.shape[0]//2 - h // 2

    return img[int(y):int(y + h), int(x):int(x + w)]


def efficient_net(input_shape, output_shape, verbose=False, loss='binary_crossentropy', activation='softmax', optimizer='adam', metrics=['accuracy']):
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

    model = models.sequential.Sequential()
    model.add(efficientnet.EfficientNetB6(
        include_top=False,
        weights=None,
        input_tensor=None,
        input_shape=input_shape,
        pooling=None,
        classes=output_shape))

    model.add(layers.Flatten())
    model.add(layers.Dense(output_shape, activation=activation))

    model.compile(
        loss=loss,
        optimizer=optimizer,
        metrics=metrics)
    if verbose:
        model.summary()
    return model, time.time() - start


def cnn_vgg16(input_shape, output_shape, verbose=False, loss='binary_crossentropy', activation='softmax', optimizer='adam', metrics=['accuracy']):
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
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
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

    gen_train = ImageDataGenerator().flow_from_dataframe(
        dataframe=df_train,
        x_col="path",
        y_col="oa",
        class_mode="raw",
        shuffle=True,
        target_size=(180, 180),
        preprocess=preprocess_zoom,
        color_mode='grayscale')

    gen_test = ImageDataGenerator().flow_from_dataframe(
        dataframe=df_test,
        x_col="path",
        y_col="oa",
        class_mode="raw",
        shuffle=False,
        target_size=(180, 180),
        preprocess=preprocess_zoom,
        color_mode='grayscale')

    return gen_train, gen_test, df_test['oa'], time.time() - start


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
            shuffle=False,
            preprocess_function=preprocess_zoom,
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


def plot_results(history, metric, label2='epoch', X_val=None, valid=None, file_name=None, x_scale=None):
    if X_val is None:
        X_val = range(len(history[metric]))
    if valid is None:
        plt.plot(X_val, history[metric])
    else:
        plt.plot(X_val[valid], history[metric][valid])
    plt.ylabel(metric)
    plt.xlabel(label2)
    if x_scale is not None:
        plt.xscale(x_scale)
    if file_name is None:
        plt.show()
    else:
        plt.savefig(file_name)
        plt.close()


def plot_roc(hist, truth, pred, file_name=None):
    fpr, tpr, _ = roc_curve(y_true=truth, y_score=pred, pos_label=1)
    # plt.plot(fpr, tpr, label="AUC")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.ylim(0, 1)
    plt.xlim(0, 1)
    # plt.show()

    auc_keras = auc(fpr, tpr)
    plt.plot(fpr, tpr, label="AUC = {}".format(round(auc_keras, 4)))
    plt.legend()
    if file_name is None:
        plt.show()
    else:
        plt.savefig(file_name)
        plt.close()


def model_test_train_VGG(df):
    global zoom
    global lr
    global ep

    print("zoom={}. learning rate={}. epochs={}".format(zoom, lr, ep))

    # Create our model
    model, ttc = mobile_net(
        (180, 180, 1),
        1,
        loss='binary_crossentropy',
        verbose=False,
        activation='sigmoid',
        optimizer=optimizers.Adam(learning_rate=lr),  # learning_rate=1e-5),
        metrics=['accuracy', metrics.AUC()])
    print("Creating Model Took {} Seconds!".format(round(ttc, 4)))

    # Get Data Generators
    train, test, truth, ttg = generate_data(df, .8, .2, 'binary')
    print("Data Generator Creation Took {} Seconds!".format(round(ttg, 4)))

    # Test model
    hist_train, hist_test, pred, ttt = train_test(model, train, test, epochs=ep)
    # print(tf.math.confusion_matrix(truth, pred))
    print("Model Training Took {} Seconds!".format(round(ttt, 4)))

    print("Hist_train keys:", hist_train.keys())
    print("Hist_train:", hist_train)
    print("Hist_test:", hist_test)

    # Plot Results
    plot_results(hist_train, 'loss', file_name="results/MobileNet_Loss_ep={}_lr={}_zoom={}.png".format(ep, lr, zoom))
    plot_results(hist_train, 'accuracy', file_name="results/MobileNet_Accuracy_ep={}_lr={}_zoom={}.png".format(ep, lr, zoom))
    plot_roc(hist_train, truth, pred, file_name="results/MobileNet_AUC_ep={}_lr={}_zoom={}.png".format(ep, lr, zoom))

    return hist_test


def main():
    verbose = 1
    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
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
        f, axarr = plt.subplots(5, 2)

        # use the created array to output your multiple images.
        axarr[0, 0].imshow(preprocess_zoom(cv2.imread(df['path'][0])[:, :, 0], 1), )
        axarr[1, 0].imshow(preprocess_zoom(cv2.imread(df['path'][0])[:, :, 0], 2))
        axarr[2, 0].imshow(preprocess_zoom(cv2.imread(df['path'][0])[:, :, 0], 3))
        axarr[3, 0].imshow(preprocess_zoom(cv2.imread(df['path'][0])[:, :, 0], 4))
        axarr[4, 0].imshow(preprocess_zoom(cv2.imread(df['path'][0])[:, :, 0], 5))
        axarr[0, 1].imshow(preprocess_zoom(cv2.imread(df['path'][0])[:, :, 0], 6))
        axarr[1, 1].imshow(preprocess_zoom(cv2.imread(df['path'][0])[:, :, 0], 7))
        axarr[2, 1].imshow(preprocess_zoom(cv2.imread(df['path'][0])[:, :, 0], 8))
        axarr[3, 1].imshow(preprocess_zoom(cv2.imread(df['path'][0])[:, :, 0], 9))
        axarr[4, 1].imshow(preprocess_zoom(cv2.imread(df['path'][0])[:, :, 0], 10))

        axarr[0, 0].set_title("1x Scale")
        axarr[1, 0].set_title("2x Scale")
        axarr[2, 0].set_title("3x Scale")
        axarr[3, 0].set_title("4x Scale")
        axarr[4, 0].set_title("5x Scale")
        axarr[0, 1].set_title("6x Scale")
        axarr[1, 1].set_title("7x Scale")
        axarr[2, 1].set_title("8x Scale")
        axarr[3, 1].set_title("9x Scale")
        axarr[4, 1].set_title("10x Scale")

        axarr[0, 0].axis('off')
        axarr[1, 0].axis('off')
        axarr[2, 0].axis('off')
        axarr[3, 0].axis('off')
        axarr[4, 0].axis('off')
        axarr[0, 1].axis('off')
        axarr[1, 1].axis('off')
        axarr[2, 1].axis('off')
        axarr[3, 1].axis('off')
        axarr[4, 1].axis('off')

        plt.show()

    global zoom
    global lr
    global ep

    lr = 1e-2  # 1e-5
    ep = 25
    zoom = 10

    values = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    acc = {'accuracy': np.zeros(len(values))}
    auc = {'AUC': np.zeros(len(values))}

    for i in range(len(values)):
        ep = values[i]
        hist = model_test_train_VGG(df)
        acc[i] = hist[1]
        auc[i] = hist[2]

    plot_results(acc, 'accuracy', file_name="results/MobileNet_Accuracy_ep_lr={}_Zoom={}.png".format(lr, zoom), label2='Learning Rate', X_val=values)
    plot_results(auc, 'AUC', file_name="results/MobileNet_AUC_ep_lr={}_Zoom={}.png".format(lr, zoom), label2='Learning Rate', X_val=values)


if __name__ == "__main__":
    main()
