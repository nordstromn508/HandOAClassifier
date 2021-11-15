"""
@author: Nicholas Nordstrom
main.py
    Main thread of execution

"""
import glob
import os
import time


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
    pass


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
    pass


if __name__ == "__main__":
    main()
