"""This is a generic parent class for Tensorflow models. This file should be stand-alone."""
import tensorflow as tf
import numpy as np
import unittest2
from abc import ABC, abstractmethod


def create_tensorboard_visualization(model_name):
    """Saves the Tensorflow graph of your model, so you can view it in a TensorBoard console."""
    print('Creating Tensorboard visualization')
    writer = tf.summary.FileWriter("/tmp/" + model_name + "/")
    writer.add_graph(tf.get_default_graph())
    return writer


def restore_model_from_save(model_var_dir, sess, var_list=None, gpu_options=None):
    """Restores all model variables from the specified directory."""

    saver = tf.train.Saver(max_to_keep=10, var_list=var_list)
    # Restore model from previous save.
    ckpt = tf.train.get_checkpoint_state(model_var_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        print("No checkpoint found!")
        return -1


def load_scope_from_save(save_dir, sess, scope):
    """Load the encoder model variables from checkpoint in save_dir.
    Store them in session sess."""
    variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
    assert len(variables) > 0
    restore_model_from_save(save_dir, sess, var_list=variables)


# class BatchGenerator:
#     def __init__(self, datas, batch_size):
#         if isinstance(datas, list):
#             self.is_only_one_data = False
#             self.datas = datas
#         else:
#             self.is_only_one_data = True
#             self.datas = [datas]
#         self.batch_size = batch_size
#
#     def generate_batches(self):
#         m = self.datas[0].shape[0]
#         num_batches = int(m / self.batch_size + 1)
#         for batch_index in range(num_batches):
#             if batch_index == num_batches - 1:
#                 real_batch_size = m - batch_index * self.batch_size
#             else:
#                 real_batch_size = self.batch_size
#
#             if real_batch_size == 0:
#                 break
#
#             batch = [np_data[self.batch_size*batch_index:self.batch_size*batch_index+real_batch_size] for np_data in self.datas]
#             if not self.is_only_one_data:
#                 yield batch
#             else:
#                 yield batch[0]


class BatchGenerator:
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle=shuffle

    def generate_batches(self):
        m = len(self.dataset)
        indices = np.arange(m)
        np.random.shuffle(indices)
        num_batches = m // self.batch_size + 1
        for i in range(num_batches):
            for j in range(self.batch_size):
                



class Dataset(ABC):
    @abstractmethod
    def __getitem__(self, index):
        # Get a single item as an index from the dataset.
        pass

    @abstractmethod
    def __len__(self):
        # Return the length of the dataset.
        pass

    def augment(self):
        # Optional: Override for data augmentation
        pass


class DictionaryDataset(Dataset):
    def __init__(self, placeholder_dict):
        self.placeholder_dict = placeholder_dict
        self.length = -1
        for feature_name in self.placeholder_dict:
            feature_length = self.placeholder_dict[feature_name].shape[0]
            if self.length == -1:
                self.length = feature_length
            else:
                assert self.length == feature_length

    def __getitem__(self, index):
        item_dict = {}
        for feature_name in self.placeholder_dict:
            item_dict[feature_name] = self.placeholder_dict[feature_name][index, :]
        return item_dict

    def __len__(self):
        return self.length


class GenericModel(ABC):
    def __init__(self, save_dir, tensorboard_name, restore_from_save=False, **kwargs):
        # Generic model which allows for automatic saving/loading, Tensorboard visualizations,
        # batch size control, session and initialization handling, and more.
        self.params = kwargs
        self.save_dir = save_dir
        self.tensorboard_name = tensorboard_name
        self.load_scopes = []
        self.input_placeholders = {}
        self.output_tensors = {}
        self.train_ops = {}
        self.build()
        create_tensorboard_visualization(self.tensorboard_name)
        self.init = tf.global_variables_initializer()
        self.sess = tf.InteractiveSession()
        self.sess.run(self.init)
        if len(tf.trainable_variables()) > 0:
            self.saver = tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=10)
        else:
            self.saver = None
        if restore_from_save:
            for each_scope in self.load_scopes:
                load_scope_from_save(self.save_dir, self.sess, each_scope)

    @abstractmethod
    def build(self):
        # Implement model and specify model placeholders and output tensors you wish to evaluate
        # using self.input_placeholders and self.output_tensors dictionaries.
        # Specify each entry as a name:tensor pair.

        pass

    def action_per_epoch(self, all_feed_dicts, output_tensor_dict, epoch_index, is_training, **kwargs):
        # Optional: Define action to take place at the end of every epoch. Can use this
        # for printing accuracy, saving statistics, etc. If is_training=False, we are using the model for
        # prediction. Check for this.
        pass

    def action_per_batch(self, all_feed_dicts, all_output_batch_dicts, epoch_index, batch_index, is_training, **kwargs):
        # Optional: Define action to take place at the end of every batch. Can use this
        # for printing accuracy, saving statistics, etc. If is_training=False, we are using the model for
        # prediction. Check for this.
        pass

    def action_before_training(self, placeholder_dict, num_epochs, output_tensor_names=None, batch_size=32, train_op_names=None, save_per_epoch=True, **kwargs):
        # Optional: Define action to take place at the beginning of training, once. This could be used to set output_tensor_names so that certain ops always
        # execute, as needed for other action functions.
        pass

    def _eval(self, dataset, num_epochs, output_tensor_names=None, batch_size=32, is_training=True, train_op_names=None, save_per_epoch=True, **kwargs):

        self.action_before_training(dataset, num_epochs, output_tensor_names=output_tensor_names,
                                    batch_size=batch_size, train_op_names=train_op_names,
                                    save_per_epoch=save_per_epoch, **kwargs)

        # placeholder_names = []
        # placeholder_numpys = []
        # for each_name in placeholder_dict:
        #     placeholder_names.append(each_name)
        #     placeholder_numpys.append(placeholder_dict[each_name])

        # Specify training ops to run
        train_op_list = []

        if is_training:
            if train_op_names is None:
                # Use all train ops
                for op_name in self.train_ops:
                    train_op_list.append(self.train_ops[op_name])
            else
                # Use specified train ops
                for op_name in train_op_names:
                    train_op_list.append(self.train_ops[op_name])

        if output_tensor_names is None:
            output_tensor_names = [name for name in self.output_tensors]

        output_tensors = [self.output_tensors[each_tensor_name] for each_tensor_name in output_tensor_names]
        output_tensor_dict = {}

        for epoch_index in range(num_epochs):
            batch_gen = BatchGenerator(placeholder_numpys, batch_size)
            all_feed_dicts = []
            all_output_batch_dicts = []

            # Train on batches
            for batch_index, placeholder_batch_numpys in enumerate(batch_gen.generate_batches()):
                # Create feed dictionary, run on batch, collected batch i/o, print, save

                feed_dict = {self.input_placeholders[placeholder_names[index]]: placeholder_batch_numpys[index] for index in range(len(placeholder_batch_numpys))}
                [output_numpy_arrays, _] = self.sess.run([output_tensors, train_op_list], feed_dict)

                output_batch_dict = {output_tensors[index]: output_numpy_arrays[index] for index in range(len(output_tensors))}

                all_feed_dicts.append(feed_dict)
                all_output_batch_dicts.append(output_batch_dict)

                self.action_per_batch(all_feed_dicts, all_output_batch_dicts, epoch_index, batch_index, is_training, **kwargs)

            # Give user option to define their own save mechanism - otherwise save
            if save_per_epoch:
                self.saver.save(self.sess, self.save_dir, global_step=epoch_index)

            # Concatenate batches to return output tensors
            for each_tensor_name, each_tensor in zip(output_tensor_names, output_tensors):
                current_tensor_batch_numpys = []
                for each_batch_dict in all_output_batch_dicts:
                    current_tensor_batch_numpys.append(each_batch_dict[each_tensor])
                if len(current_tensor_batch_numpys[0].shape) > 0:
                    output_tensor_dict[each_tensor_name] = np.concatenate(current_tensor_batch_numpys, axis=0)
                else:
                    output_tensor_dict[each_tensor_name] = np.mean(current_tensor_batch_numpys)

            self.action_per_epoch(dataset, output_tensor_dict, epoch_index, is_training, **kwargs)

        return output_tensor_dict

    def predict(self, placeholder_dict, output_tensor_names=None, batch_size=32):

        output_tensor_dict = self._eval(placeholder_dict, 1,
                                        output_tensor_names=output_tensor_names,
                                        batch_size=batch_size,
                                        train_op_names=[],
                                        save_per_epoch=False)

        return output_tensor_dict


class SimpleModel(GenericModel):
    def build(self):
        tf_input = tf.placeholder(tf.float32, shape=(None, 1), name='x')
        tf_output = tf_input + 3.0
        self.input_placeholders['x'] = tf_input
        self.output_tensors['y'] = tf_output

    def action_per_epoch(self, all_feed_dicts, output_tensor_dict, epoch_index, is_training, **kwargs):
        print('Executing action_per_epoch')

    def action_per_batch(self, all_feed_dicts, all_output_batch_dicts, epoch_index, batch_index, is_training, **kwargs):
        print('Executing action_per_batch')


class LessSimpleModel(GenericModel):
    def build(self):
        tf_input = tf.placeholder(tf.float32, shape=(None, 1), name='x')
        tf_w = tf.get_variable('w', (1, 1), initializer=tf.contrib.layers.xavier_initializer())
        tf_output = tf_input + tf_w
        self.input_placeholders['x'] = tf_input
        self.output_tensors['y'] = tf_output
        self.output_tensors['w'] = tf_w

        tf_label = tf.placeholder(tf.float32, shape=(None, 1), name='label')
        tf_loss = tf.nn.l2_loss(tf_label - self.output_tensors['y'])
        train_op = tf.train.AdamOptimizer(.001).minimize(tf_loss)
        self.input_placeholders['label'] = tf_label
        self.output_tensors['loss'] = tf_loss
        self.train_ops['l2_loss'] = train_op


class GenericModelTest(unittest2.TestCase):
    def setUp(self):
        pass

    def test_simple_model_creation(self):
        SimpleModel('/tmp/sm_save/', 'sm')

    def test_simple_model_prediction(self):
        sm = SimpleModel('/tmp/sm_save/', 'sm')

        output_dict = sm.predict({'x': np.array([[3], [4]])})

        assert np.array_equal(output_dict['y'], np.array([[6.], [7.]]))

    def test_simple_model_train(self):
        sm = SimpleModel('/tmp/sm_save/', 'sm')

        d = DictionaryDataset({'x': np.array([[3], [4]])})

        output_dict = sm._eval(d, num_epochs=10, save_per_epoch=False)

        assert np.array_equal(output_dict['y'], np.array([[6.], [7.]]))

    def test_less_simple_model_train(self):
        lsm = LessSimpleModel('/tmp/lsm_save/', 'lsm')

        output_dict = lsm._eval({'x': np.array([[3]]), 'label': np.array([[6]])}, num_epochs=10000, save_per_epoch=False)
        epsilon = .01
        assert np.abs(3 - output_dict['w']) < epsilon
        print(output_dict)

        y = lsm.predict({'x': np.array([[7]])}, ['y'])['y']
        assert np.abs(y - 10) < epsilon
