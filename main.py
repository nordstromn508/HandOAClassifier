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
from keras import models, layers, datasets


def get_data(n=None):
    """
    method to import data and get it all nice and ready for learning.
    :param n: number of image sets to obtain
    :return: our dataset.
    """
    start = time.time()
    if n is None:
        images = [glob.glob("data/FingerJoints/" + x[0:7] + "*") for x in os.listdir("data/FingerJoints/")[0::12]]
    else:
        images = [glob.glob("data/FingerJoints/" + x[0:7] + "*") for x in os.listdir("data/FingerJoints/")[0:12*n:12]]
    return images, time.time()-start

def get_label(data):

    """
    method to get the label of the data.
    :param data: data to get label of.
    :return: label of the data.
    """

    file = pd.read_excel(data)
    file = file.values[:,:12]
    list = []
    for i in range(len(file)):
        for j in range(len(file[i])):
            if file[i][j] == 'nan':
                file[i][j] = 0
            
            list.append(file[i][j])

    return np.array(list)
    


def preprocess(data):
    """
    preprocesses our data so that it is more easily understood by our algorithm.
    :return: preprocessed data.
    """
    pass
    


def data_augment(data):
    """
    performs data augmentation as needed.
    :param data: data to augment.
    :return: augmented form of the data.
    """
    pass


def create_cnn():
    """
    creates our algorithm to learn from our dataset.
    :return: the model object.
    """
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(180, 180, 1)))
    model.add(layers.AveragePooling2D((2, 2)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(90, 90, 1)))
    model.add(layers.AveragePooling2D((2, 2)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(45, 45, 1)))
    model.add(layers.AveragePooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(2, activation='softmax'))

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model


def pipeline():
    """
    This is our data 'pipeline' it ensures a consistent flow and modulation of the data.
    :return: None.
    """
    pass


def main():
    paths, ttr = get_data(50)
    print(paths[0])
    print("Getting data took {} seconds!".format(ttr))

    paths = ["data/FingerJoints/9000099_dip5.png", "data/FingerJoints/9000099_dip4.png", "data/FingerJoints/9000099_dip3.png", "data/FingerJoints/9000099_dip2.png"]
    X = []
    for p in paths:
        X.append(cv2.imread(p)[:, :, 0])
    X = np.array(X)
    y = np.ones(X.shape[0])

    print(X.shape)
    print(y.shape)

    model = create_cnn()
    model.fit(X, y)
    pass


if __name__ == "__main__":
    main()
