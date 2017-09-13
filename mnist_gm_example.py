"""GenericModel class is used to train an MNIST classifier based on the Tensorflow MNIST tutorial.
Create MNIST model, add per-batch action to print the batch number. Train it and then calculate
training and testing accuracies."""
import generic_model
import tensorflow as tf
import numpy as np

def example_basic_mnist():
    class MNISTModel(generic_model.GenericModel):
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
            self.input_placeholders['image'] = x
            self.input_placeholders['label'] = y_
            self.output_tensors['prediction'] = y
            self.output_tensors['accuracy'] = correct_prediction

    mnist_model = MNISTModel()

    mnist_train_set = generic_model.MNISTTrainSet()
    mnist_test_set = generic_model.MNISTTestSet()

    train_dict = mnist_model.train(mnist_train_set, num_epochs=1)

    #train_dict = mnist_model.predict(mnist_train_set, ['accuracy'])
    print('Training accuracy: %s' % np.mean(train_dict['accuracy']))

    test_dict = mnist_model.predict(mnist_test_set, ['accuracy'])
    print('Test accuracy: %s' % np.mean(test_dict['accuracy']))


def example_advanced_mnist():
    class ConvMNISTModel(generic_model.GenericModel):
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
            self.input_placeholders['image'] = x
            self.input_placeholders['label'] = y_
            self.input_placeholders['keep prob'] = keep_prob
            self.output_tensors['prediction'] = y_conv
            self.output_tensors['accuracy'] = correct_prediction

        def action_per_batch(self, input_batch_dict, output_batch_dict, epoch_index, batch_index, is_training, **kwargs):
            if is_training:
                if batch_index % 1000 == 0:
                    print('Batch: %s' % batch_index)
                    print('Accuracy: %s' % np.mean(output_batch_dict['accuracy']))

    mnist_model = ConvMNISTModel()

    mnist_train_set = generic_model.MNISTTrainSet(num_examples=20000 * 50)
    mnist_test_set = generic_model.MNISTTestSet()

    train_dict = mnist_model.train(mnist_train_set, num_epochs=1, parameter_dict={'keep prob': .5})

    #train_dict = mnist_model.predict(mnist_train_set, ['accuracy'])
    print('Training accuracy: %s' % np.mean(train_dict['accuracy']))

    test_dict = mnist_model.predict(mnist_test_set, ['accuracy'], parameter_dict={'keep prob': 1.0})
    print('Test accuracy: %s' % np.mean(test_dict['accuracy']))


print('Basic MNIST model from Tensorflow tutorials')

example_basic_mnist()

print()
print('Advanced MNIST model from Tensorflow tutorials')

example_advanced_mnist()

