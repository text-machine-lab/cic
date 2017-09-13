"""GenericModel class is used to train an MNIST classifier based on the Tensorflow MNIST tutorial.
Create MNIST model, add per-batch action to print the batch number. Train it and then calculate
training and testing accuracies."""
import generic_model
import tensorflow as tf
import numpy as np


class MNISTModel(generic_model.GenericModel):
    def build(self):
        """Copied from Tensorflow tutorial simple MNIST example."""
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

    def action_per_batch(self, input_batch_dict, output_batch_dict, epoch_index, batch_index, is_training, **kwargs):
        if is_training:
            print('Batch: %s' % batch_index)

mnist_model = MNISTModel()

mnist_train_set = generic_model.MNISTTrainSet()
mnist_test_set = generic_model.MNISTTestSet()

train_dict = mnist_model.train(mnist_train_set, num_epochs=1, parameter_dict={'keep prob': .5})

#train_dict = mnist_model.predict(mnist_train_set, ['accuracy'])
print('Training accuracy: %s' % np.mean(train_dict['accuracy']))

test_dict = mnist_model.predict(mnist_test_set, ['accuracy'], parameter_dict={'keep prob': 1.0})
print('Test accuracy: %s' % np.mean(test_dict['accuracy']))

