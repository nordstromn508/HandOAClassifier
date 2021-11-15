"""
@author: Nicholas Nordstrom
main.py
    Main thread of execution

"""
import cv2
import numpy as np
import tensorflow as tf
from keras import models, layers, datasets


def get_data():
    """
    method to import data and get it all nice and ready for learning.
    :return: our dataset.
    """
    pass


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
