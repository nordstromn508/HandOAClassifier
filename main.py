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
from keras import models, layers


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
            label.append((file.at[id, joint]))

    label = np.array(label, dtype=object)
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


def create_cnn():
    """
    creates our algorithm to learn from our dataset.
    :return: the model object.
    """
    start = time.time()
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
    return model, time.time() - start


def pipeline(paths, y):
    """
    This is our data 'pipeline' it ensures a consistent flow and modulation of the data. implements lazy image loading.
    :return:
    """
    start = time.time()
    X = [[cv2.imread(p)[:, :, 0] for p in path] for path in paths]
    X, ttt = preprocess(X)
    X_aug, tta = data_augment(X)
    y_aug = [_ for _ in y]
    return X, y, X_aug, y_aug, ttt, tta, time.time() - start


def main():
    # Get chunk of data
    paths, y, ttr = get_data(50)
    print("Getting data paths took {} seconds!".format(ttr))

    # Create our model
    model, ttc = create_cnn()
    print("Creating model took {} seconds!".format(ttc))

    # Send data to the pipeline
    X, y, X_aug, y_aug, ttt, tta, ttp = pipeline(paths, y)
    print("Data transformation took {} seconds!".format(ttt))
    print("Data augmentation took {} seconds!".format(tta))
    print("Total data pipeline took {} seconds!".format(ttp))

    # Testing Data Shape
    print("X len:", len(X))
    print("X[0] len:", len(X[0]))
    print("X[0][0] len:", len(X[0][0]))
    print("x[0][0]:", X[0][0])
    print("X[0][0][0] len:", len(X[0][0][0]))
    print("X[0][0][0][0]:", X[0][0][0][0])

    # Train model on training data
    # model.fit(X[0][0], y)


if __name__ == "__main__":
    main()
