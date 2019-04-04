import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math
import h5py
import random
import copy
import sys


# Some Helper Functions
def load_data(dataset, standardize=True):
    """
    :param dataset:
    :param standardize:
    :return:
    """

    features = dataset['arr'][:, 0]
    features = np.array([feature for feature in features])
    features = np.reshape(features, (features.shape[0], features.shape[1] * features.shape[2]))

    if standardize:
        features = StandardScaler().fit_transform(features)

    labels = dataset['arr'][:, 1]
    labels = np.array([label for label in labels])

    return features, labels

def toBinaryArray(imgArray):
    """ 
    This function takes in one greyscale image representing an executable, and returns the
    binary representation.
    """
    result = []
    for pixel in imgArray:
        temp = str(bin(int(pixel)))[2:]
        result.append("0"*(8-len(temp))+temp)
    return result

def toImgArray(binaryArray):
    """
    This function takes in one binary executable (in array format), and retursn the
    pixel representation.
    """
    result = []
    for byte in binaryArray:
        result.append(int(byte, 2))
    return result


def hexToBinary(hexByte):
    """ 
    Turn hex bytes into binary string.
    """
    temp = "{0:8b}".format(int(hexByte,16)).strip()
    temp = "0"*(8-len(temp)) + temp
    return temp

def hexToBinaryArray(string):
    """
    This function takes in a continuous string of hex values and translates to array of bytes. 
    e.g., "000" -> ['00000000', '00000000', '00000000']
    """
    result = []
    n = 2
    temp = [string[i:i+n] for i in range(0, len(string), n)]
    for byte in temp:
        result.append(hexToBinary(byte))
    return result

def binaryToHexArray(binArray):
    """
    
    """
    temp = []
    for byte in binArray:
        temp2 = str(hex(int(byte, 2)))
        if len(temp2) < 4:
            temp.append("0" + temp2[2:])
        else:
            temp.append(str(hex(int(byte, 2)))[2:])
    return temp

def arrayToString(array):
    """
    
    """
    return " ".join(array)

def slist(string):
    return [string[i:i+2] for i in range(0, len(string), 2)]

def plot_images(images, cls_true, cls_pred=None, noise=0.0):
    assert len(images) == len(cls_true) == 9
    
    # Create figure with 3x3 subplots
    fig, axes = plt.subplots(3,3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    
    for i, ax in enumerate(axes.flat):
        # Get the i'th image, and reshape the array.
        image = images[i].reshape((40, 40))
        
        # Add the adversarial noise to the image.
        image += noise
        
        # Ensure the noisy pixel-values are between 0 and 1.
        image = np.clip(image, 0., 255.)
        
        # Plot the image.
        ax.imshow(image,
                  cmap='gray_r')
        
        # Show true and predicted classes.
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])
            
        # Show the classes as the label on the x_axis.
        ax.set_xlabel(xlabel)
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
        
    # End for-loop
    plt.show()