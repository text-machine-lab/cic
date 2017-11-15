from gemtk.dataset import DictionaryDataset
from tensorflow.examples.tutorials.mnist import input_data

class MNISTTrainSet(DictionaryDataset):
    """Create dataset of MNIST examples for training.

    Features:
        'image': numpy array of 784 pixel images (flattened from 28 x 28)
        'label': numpy array of labels 0 - 9 indicating digit contained in each image"""
    def __init__(self):
        self.mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
        DictionaryDataset.__init__(self, {'image': self.mnist.train.images, 'label': self.mnist.train.labels})


class MNISTTestSet(DictionaryDataset):
    """Create dataset of MNIST examples for testing.

    Features:
        'image': numpy array of 784 pixel images (flattened from 28 x 28)
        'label': numpy array of labels 0 - 9 indicating digit contained in each image"""

    def __init__(self):
        self.mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
        DictionaryDataset.__init__(self, {'image': self.mnist.test.images, 'label': self.mnist.test.labels})