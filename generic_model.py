"""This is a generic parent class for Tensorflow models. This file should be stand-alone."""
import tensorflow as tf
import numpy as np
from abc import ABC, abstractmethod


def create_tensorboard_visualization(model_name):
    """Saves the Tensorflow graph of your model, so you can view it in a TensorBoard console."""
    print('Creating Tensorboard visualization')
    writer = tf.summary.FileWriter("/tmp/" + model_name + "/")
    writer.add_graph(tf.get_default_graph())


def restore_model_from_save(model_var_dir, var_list=None, sess=None, gpu_options=None):
    """Restores all model variables from the specified directory."""
    if sess is None:
        sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))

    saver = tf.train.Saver(max_to_keep=10, var_list=var_list)
    # Restore model from previous save.
    ckpt = tf.train.get_checkpoint_state(model_var_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        print("No checkpoint found!")
        return -1

    return sess


def load_scope_from_save(save_dir, sess, scope):
    """Load the encoder model variables from checkpoint in save_dir.
    Store them in session sess."""
    variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
    assert len(variables) > 0
    restore_model_from_save(save_dir, var_list=variables, sess=sess)


class BatchGenerator:
    def __init__(self, datas, batch_size):
        if isinstance(datas, list):
            self.datas = datas
        else:
            self.datas = [datas]
        self.batch_size = batch_size

    def generate_batches(self):
        m = self.datas[0].shape[0]
        num_batches = int(m / self.batch_size + 1)
        for batch_index in range(num_batches):
            if batch_index == num_batches - 1:
                real_batch_size = m - batch_index * self.batch_size
            else:
                real_batch_size = self.batch_size

            if real_batch_size == 0:
                break

            batch = [np_data[self.batch_size*batch_index:self.batch_size*batch_index+real_batch_size] for np_data in self.datas]
            if len(batch) > 1:
                yield batch
            else:
                yield batch[0]


class GenericModel(ABC):
    def __init__(self, save_dir, restore_from_save=False, params=None):
        # Here, inputs means placeholders and outputs means tensors of interest
        self.params = params
        self.model_inputs, self.model_outputs, self.load_scopes = self.build()
        self.train_ops = self.build_trainer()
        create_tensorboard_visualization(self.__name__)
        self.init = tf.global_variables_initializer()
        self.sess = tf.InteractiveSession()
        self.sess.run(self.init)
        if restore_from_save:
            load_scope_from_save(save_dir, self.sess, self.load_scopes)

    @abstractmethod
    def build(self):
        pass

    @abstractmethod
    def build_trainer(self):
        pass

    def train(self, inputs, outputs, labels, num_epochs, batch_size):
        ### NOT IMPLEMENTED!!!!
        for epoch in range(num_epochs):
            print('Epoch: %s' % epoch)
            per_print_losses = []
            batch_gen = BatchGenerator([np_latent_message, np_latent_response], batch_size)

            for np_message_batch, np_response_batch in latent_batch_gen.generate_batches():
                assert np_message_batch.shape[0] != 0
                np_batch_loss, np_batch_response, _ = self.sess.run([self.tf_total_loss, self.tf_latent_prediction, self.train_op],
                                                               feed_dict={self.tf_latent_message: np_message_batch,
                                                                          self.tf_latent_response: np_response_batch,
                                                                          self.tf_keep_prob: keep_prob})
                per_print_losses.append(np_batch_loss)

            print('Message std: %s' % np.std(np_message_batch))
            print('Response std: %s' % np.std(np_response_batch))
            print('Prediction std: %s' % np.std(np_batch_response))
            print('Loss: %s' % np.mean(per_print_losses))
            self.saver.save(self.sess, self.save_dir, global_step=epoch)

    def predict(self, inputs, outputs, batch_size):
        pass

