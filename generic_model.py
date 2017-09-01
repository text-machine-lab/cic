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


def concatenate_batch_dictionaries(batch_dictionaries, single_examples=False):
    # Concatenates numpy dictionaries. If numpy arrays represent single examples (no batch axis),
    # set single_examples=True. Otherwise false.
    # batch_dictionaries - list of dictionaries, all containing identical keys, each key being
    # a feature name
    result = {}

    for key in batch_dictionaries[0]:
        if single_examples or len(batch_dictionaries[0][key].shape) == 0:
            result[key] = np.stack([d[key] for d in batch_dictionaries], axis=0)
        else:
            result[key] = np.concatenate([d[key] for d in batch_dictionaries], axis=0)

    return result


class Dataset(ABC):
    @abstractmethod
    def __getitem__(self, index):
        # Get a single item as an index from the dataset.
        pass

    @abstractmethod
    def __len__(self):
        # Return the length of the dataset.
        pass

    def generate_batches(self, batch_size, shuffle=False):
        # Yield one batch from the dataset
        m = len(self)
        indices = np.arange(m)
        if shuffle:
            np.random.shuffle(indices)
        num_batches = m // batch_size + 1

        for i in range(num_batches):
            index_batch = indices[i * batch_size:i * batch_size+batch_size]
            if len(index_batch) == 0:
                break

            batch_data = [self[each_index] for each_index in index_batch]
            # result = {}
            # for key in batch_data[0]:
            #     result[key] = np.stack([d[key] for d in batch_data], axis=0)

            yield concatenate_batch_dictionaries(batch_data, single_examples=True)


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

        if self.length <= 0:
            raise ValueError('Cannot have zero-length dataset.')

    def __getitem__(self, index):
        item_dict = {}
        for feature_name in self.placeholder_dict:
            item_dict[feature_name] = self.placeholder_dict[feature_name][index]
        return item_dict

    def __len__(self):
        return self.length

    def to_dict(self):
        return self.placeholder_dict


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

    def action_per_epoch(self, output_tensor_dict, epoch_index, is_training, **kwargs):
        # Optional: Define action to take place at the end of every epoch. Can use this
        # for printing accuracy, saving statistics, etc. If is_training=False, we are using the model for
        # prediction. Check for this. Returns true to continue training. Only return false if you wish to
        # implement early-stopping.
        return True

    def action_per_batch(self, input_batch_dict, output_batch_dict, epoch_index, batch_index, is_training, **kwargs):
        # Optional: Define action to take place at the end of every batch. Can use this
        # for printing accuracy, saving statistics, etc. If is_training=False, we are using the model for
        # prediction. Check for this.
        pass

    def action_before_training(self, placeholder_dict, num_epochs, is_training, output_tensor_names=None, batch_size=32, train_op_names=None, save_per_epoch=True, **kwargs):
        # Optional: Define action to take place at the beginning of training, once. This could be used to set
        # output_tensor_names so that certain ops always execute, as needed for other action functions.
        # Returns whether model will auto-save after each epoch, using the caller preference as default. Only
        # set this to false if you want to handle saving in a custom way.
        return save_per_epoch

    def _eval(self, dataset, num_epochs, output_tensor_names=None, batch_size=32, is_training=True, train_op_names=None, save_per_epoch=True, **kwargs):

        save_per_epoch = self.action_before_training(dataset, num_epochs, is_training,
                                                     output_tensor_names=output_tensor_names,
                                                     batch_size=batch_size, train_op_names=train_op_names,
                                                     save_per_epoch=save_per_epoch, **kwargs)

        # Control what train ops are executed via arguments
        if is_training and train_op_names is None:
            train_op_list = [self.train_ops[op_name] for op_name in self.train_ops]
        else:
            train_op_list = [self.train_ops[op_name] for op_name in train_op_names]

        # If user doesn't specify output tensors, evaluate them all!
        if output_tensor_names is None:
            output_tensor_names = [name for name in self.output_tensors]

        # Create list of output tensors, initialize output dictionaries
        output_tensors = [self.output_tensors[each_tensor_name] for each_tensor_name in output_tensor_names]
        all_output_batch_dicts = None

        # Evaluate and/or train on dataset. Run user-defined action functions
        for epoch_index in range(num_epochs):
            all_output_batch_dicts = []
            for batch_index, batch_dict in enumerate(dataset.generate_batches(batch_size=batch_size, shuffle=is_training)):
                # Run batch in session
                feed_dict = {self.input_placeholders[feature_name]: batch_dict[feature_name] for feature_name in batch_dict}
                output_numpy_arrays, _ = self.sess.run([output_tensors, train_op_list], feed_dict)

                input_batch_dict = {feature_name: feed_dict[self.input_placeholders[feature_name]] for feature_name in batch_dict}
                output_batch_dict = {output_tensor_names[index]: output_numpy_arrays[index] for index in range(len(output_tensor_names))}

                # Keep history of batch inputs/outputs
                all_output_batch_dicts.append(output_batch_dict)

                self.action_per_batch(input_batch_dict, output_batch_dict, epoch_index, batch_index, is_training, **kwargs)

            if save_per_epoch:
                self.saver.save(self.sess, self.save_dir, global_step=epoch_index)

            continue_training = self.action_per_epoch(all_output_batch_dicts, epoch_index, is_training, **kwargs)
            if not continue_training:
                break

        # Calculate output datasets
        output_dict_concat = concatenate_batch_dictionaries(all_output_batch_dicts)

        return output_dict_concat

    def train(self, dataset, num_epochs, output_tensor_names=None, batch_size=32, **kwargs):

        output_tensor_dict = self._eval(dataset, num_epochs,
                                        output_tensor_names=output_tensor_names,
                                        batch_size=batch_size,
                                        train_op_names=None,
                                        is_training=True,
                                        save_per_epoch=True)

        return output_tensor_dict

    def predict(self, dataset, output_tensor_names=None, batch_size=32):

        output_tensor_dict = self._eval(dataset, 1,
                                        output_tensor_names=output_tensor_names,
                                        batch_size=batch_size,
                                        train_op_names=[],
                                        is_training=False,
                                        save_per_epoch=False)

        return output_tensor_dict


class SimpleModel(GenericModel):
    def build(self):
        tf_input = tf.placeholder(tf.float32, shape=(None, 1), name='x')
        tf_output = tf_input + 3.0
        self.input_placeholders['x'] = tf_input
        self.output_tensors['y'] = tf_output

    def action_per_epoch(self, output_tensor_dict, epoch_index, is_training, **kwargs):
        print('Executing action_per_epoch')

    def action_per_batch(self, input_batch_dict, output_batch_dict, epoch_index, batch_index, is_training, **kwargs):
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

    def action_before_training(self, placeholder_dict, num_epochs, is_training, output_tensor_names=None, batch_size=32, train_op_names=None, save_per_epoch=True, **kwargs):
        return False  # Do not save after each epoch


class GenericModelTest(unittest2.TestCase):
    def setUp(self):
        pass

    def test_batch_generator_and_dictionary_dataset_arange(self):
        # non-shuffled batches correspond to input feature dictionary.
        data = np.random.uniform(size=(50, 3))
        feature_dict = {'f1': data[:, 0], 'f2': data[:, 1], 'f3': data[:, 2]}
        dts = DictionaryDataset(feature_dict)
        batch_size = 10
        for index, batch_dict in enumerate(dts.generate_batches(batch_size=batch_size, shuffle=False)):
            for feature in batch_dict:
                #print('batch_dict: ' + str(batch_dict[feature]))
                #print('feature_dict: ' + str(feature_dict[feature][index*batch_size:index*batch_size+batch_size]))
                assert np.array_equal(batch_dict[feature], feature_dict[feature][index*batch_size:index*batch_size+batch_size])

    def test_batch_generator_and_dictionary_dataset_shuffle(self):
        # Set shuffle to true and batches will not correspond to input feature dictionary
        data = np.random.uniform(size=(50, 3))
        feature_dict = {'f1': data[:, 0], 'f2': data[:, 1], 'f3': data[:, 2]}
        dts = DictionaryDataset(feature_dict)
        batch_size = 10
        for index, batch_dict in enumerate(dts.generate_batches(batch_size=batch_size, shuffle=True)):
            for feature in batch_dict:
                #print('batch_dict: ' + str(batch_dict[feature]))
                #print('feature_dict: ' + str(feature_dict[feature][index*batch_size:index*batch_size+batch_size]))
                assert np.not_equal(batch_dict[feature], feature_dict[feature][index*batch_size:index*batch_size+batch_size]).any()

    def test_batch_generator_and_dictionary_dataset_arange_feature_vectors(self):
        data = np.random.uniform(size=(50, 5))
        feature_dict = {'f1': data[:, 0:2], 'f2': data[:, 2:4], 'f3': data[:, 4]}
        dts = DictionaryDataset(feature_dict)
        batch_size = 10
        for index, batch_dict in enumerate(dts.generate_batches(batch_size=batch_size, shuffle=False)):
            for feature in batch_dict:
                #print('batch_dict: ' + str(batch_dict[feature]))
                #print('feature_dict: ' + str(feature_dict[feature][index*batch_size:index*batch_size+batch_size]))
                assert np.array_equal(batch_dict[feature], feature_dict[feature][index*batch_size:index*batch_size+batch_size])

    def test_batch_generator_and_dictionary_dataset_remainder(self):
        data = np.random.uniform(size=(10, 3))
        feature_dict = {'f1': data[:, 0], 'f2': data[:, 1], 'f3': data[:, 2]}
        dts = DictionaryDataset(feature_dict)
        batch_size = 20
        for batch_dict in dts.generate_batches(batch_size=batch_size, shuffle=False):
            assert batch_dict['f1'].shape == batch_dict['f2'].shape
            assert batch_dict['f2'].shape == batch_dict['f3'].shape
            assert batch_dict['f1'].shape[0] == 10

    def test_batch_generator_and_empty_dictionary_dataset(self):
        with self.assertRaises(ValueError):
            DictionaryDataset({})

    def test_dictionary_dataset(self):
        data = np.random.uniform(size=(10, 3))
        feature_dict = {'f1': data[:, 0], 'f2': data[:, 1], 'f3': data[:, 2]}
        dts = DictionaryDataset(feature_dict)
        for feature_name in feature_dict:
            feature = feature_dict[feature_name]
            for index in range(feature.shape[0]):
                example_value = feature[index]
                dataset_example = dts[index]
                assert isinstance(dataset_example, dict)
                #print(dataset_example)
                dataset_example_value = dataset_example[feature_name]
                assert np.array_equal(example_value, dataset_example_value)

    def test_dictionary_dataset_vector_features(self):
        data = np.random.uniform(size=(10, 5))
        feature_dict = {'f1': data[:, 0:2], 'f2': data[:, 2:4], 'f3': data[:, 4]}
        dts = DictionaryDataset(feature_dict)
        for feature_name in feature_dict:
            feature = feature_dict[feature_name]
            for index in range(feature.shape[0]):
                example_value = feature[index]
                dataset_example = dts[index]
                assert isinstance(dataset_example, dict)
                #print(dataset_example)
                dataset_example_value = dataset_example[feature_name]
                assert np.array_equal(example_value, dataset_example_value)

    def test_simple_model_creation(self):
        SimpleModel('/tmp/sm_save/', 'sm')

    def test_simple_model_prediction(self):
        sm = SimpleModel('/tmp/sm_save/', 'sm')

        dataset = DictionaryDataset({'x': np.array([[3], [4]])})

        output_dict = sm.predict(dataset)

        print(output_dict)

        assert np.array_equal(output_dict['y'], np.array([[6.], [7.]]))

    def test_simple_model_train(self):
        sm = SimpleModel('/tmp/sm_save/', 'sm')

        d = DictionaryDataset({'x': np.array([[3], [4]])})

        output_dict = sm.train(d, num_epochs=10)

        print(output_dict)

        # Account for automatic shuffling - check if outputs are correct.
        assert np.array_equal(output_dict['y'], np.array([[6.], [7.]])) or \
               np.array_equal(output_dict['y'], np.array([[7.], [6.]]))

    def test_less_simple_model_train(self):
        lsm = LessSimpleModel('/tmp/lsm_save/', 'lsm')

        dataset = DictionaryDataset({'x': np.array([[3]]), 'label': np.array([[6]])})

        output_dict = lsm.train(dataset, num_epochs=10000)
        epsilon = .01
        assert np.abs(3 - output_dict['w']) < epsilon
        print(output_dict)

        output_dict = lsm.predict(dataset, ['y'])

        print(output_dict['y'])

        assert np.abs(output_dict['y'] - 6) < epsilon
