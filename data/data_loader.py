import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical

def load_mnist_data():
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    
    train_images = train_images.reshape((60000, 28 * 28)).astype('float32') / 255
    test_images = test_images.reshape((10000, 28 * 28)).astype('float32') / 255
    
    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)
    
    return (train_images, train_labels), (test_images, test_labels)
