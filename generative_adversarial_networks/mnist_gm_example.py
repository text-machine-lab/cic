"""GenericModel class is used to train an MNIST classifier based on the Tensorflow MNIST tutorial.
Create MNIST model, add per-batch action to print the batch number. Train it and then calculate
training and testing accuracies."""
import timeit

import numpy as np
import tensorflow as tf

import gemtk


def example_basic_mnist():
    class MNISTModel(gemtk.GenericModel):
        def build(self):
            """Copied from Tensorflow tutorial MNIST for ML Beginners example."""
            x = tf.placeholder(tf.float32, [None, 784])
            w = tf.Variable(tf.zeros([784, 10]))
            b = tf.Variable(tf.zeros([10]))
            y = tf.nn.softmax(tf.matmul(x, w) + b)
            y_ = tf.placeholder(tf.float32, [None, 10])
            self.loss = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

            # Create interface
            self.inputs['image'] = x
            self.inputs['label'] = y_
            self.outputs['prediction'] = y
            self.outputs['accuracy'] = correct_prediction

    mnist_model = MNISTModel()

    mnist_train_set = gemtk.MNISTTrainSet()
    mnist_test_set = gemtk.MNISTTestSet()

    train_dict = mnist_model.train(mnist_train_set, num_epochs=1)

    #train_dict = mnist_model.predict(mnist_train_set, ['accuracy'])
    print('Training accuracy: %s' % np.mean(train_dict['accuracy']))

    test_dict = mnist_model.predict(mnist_test_set, ['accuracy'])
    print('Test accuracy: %s' % np.mean(test_dict['accuracy']))


def example_advanced_mnist_batch():
    example_advanced_mnist(use_batch_mnist_dataset=True)


def example_advanced_mnist(use_batch_mnist_dataset=False):
    class ConvMNISTModel(gemtk.GenericModel):
        def build(self):
            """Copied from Tensorflow tutorial Deep MNIST example."""
            def weight_variable(shape):
                initial = tf.truncated_normal(shape, stddev=0.1)
                return tf.Variable(initial)

            def bias_variable(shape):
                initial = tf.constant(0.1, shape=shape)
                return tf.Variable(initial)

            def conv2d(x, W):
                return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

            def max_pool_2x2(x):
                return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                                      strides=[1, 2, 2, 1], padding='SAME')

            x = tf.placeholder(tf.float32, [None, 784])

            W_conv1 = weight_variable([5, 5, 1, 32])
            b_conv1 = bias_variable([32])

            x_image = tf.reshape(x, [-1, 28, 28, 1])

            h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
            h_pool1 = max_pool_2x2(h_conv1)

            W_conv2 = weight_variable([5, 5, 32, 64])
            b_conv2 = bias_variable([64])

            h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
            h_pool2 = max_pool_2x2(h_conv2)

            W_fc1 = weight_variable([7 * 7 * 64, 1024])
            b_fc1 = bias_variable([1024])

            h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
            h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

            keep_prob = tf.placeholder(tf.float32)
            h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

            W_fc2 = weight_variable([1024, 10])
            b_fc2 = bias_variable([10])

            y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
            y_ = tf.placeholder(tf.float32, [None, 10])

            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
            correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))

            # Create interface
            self.inputs['image'] = x
            self.inputs['label'] = y_
            self.inputs['keep prob'] = keep_prob
            self.outputs['prediction'] = y_conv
            self.outputs['accuracy'] = correct_prediction

    mnist_model = ConvMNISTModel()

    if use_batch_mnist_dataset:
        mnist_train_set = gemtk.BatchMNISTTrainSet()
    else:
        mnist_train_set = gemtk.MNISTTrainSet()

    mnist_test_set = gemtk.MNISTTestSet()

    def train():
        train_dict = mnist_model.train(mnist_train_set, num_epochs=10, parameter_dict={'keep prob': .5})

        print('Training accuracy: %s' % np.mean(train_dict['accuracy']))

    print('Execution time: %s' % timeit.timeit(train, number=1))

    test_dict = mnist_model.predict(mnist_test_set, ['accuracy'], parameter_dict={'keep prob': 1.0})
    print('Test accuracy: %s' % np.mean(test_dict['accuracy']))

if __name__ == '__main__':

    print('Basic MNIST model from Tensorflow tutorials')

    example_basic_mnist()

    print()
    print('Advanced MNIST model from Tensorflow tutorials')

    print('Timing using dataset without overloaded batching...')
    example_advanced_mnist(use_batch_mnist_dataset=False)
    print('Timing using dataset with overloaded batching...')
    example_advanced_mnist(use_batch_mnist_dataset=True)
    print('Similar execution times indicate non-batching dataset performs well')
    print('The standard dataset format is non-batching (overloaded __getitem__ operator)')



