"""This is a generic parent class for Tensorflow models. This file should be stand-alone."""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import unittest2
import os
from abc import ABC, abstractmethod


class GenericModel(ABC):
    def __init__(self, save_dir=None, tensorboard_name=None, restore_from_save=False, trainable=True, tf_log_level='2'):
        """Abstract class which contains support functionality for developing Tensorflow models.
        Derive subclasses from this class and override the build() method. Define entire model in this method,
        and add all placeholders to self.input_placeholders dictionary as name:placeholder pairs. Add all tensors
        you wish to evaluate to self.output_tensors as name:tensor pairs. Assign loss function you wish to train on
        to self.loss variable ('loss' tensor automatically added to self.output_tensors). Names chosen for placeholders
        and output tensors are used to refer to these tensors outside of the model. As a general rule, Tensorflow
        should not need to be imported by the caller of this object. Once model is defined, training and prediction are
        already implemented. Model graph is automatically saved to Tensorboard directory. Model parameters are
        automatically saved after each epoch, and restored from save after training if desired. If restored,
        entire model can be loaded, or only specific scopes by adding scopes to self.load_scopes list. Variable
        initialization and session maintenance are handled internally. This model supports custom operations
        if necessary.

        Arguments:
            save_dir: directory with which to save your model checkpoint files to
            tensorboard_name: directory name in /tmp/ where Tensorboard graph will be saved
            restore_from_save: indicates whether to restore model from save
            trainable: decide whether model will be trainable
            tf_log_level: by default, disables all outputs from Tensorflow backend (except errors)
        """
        assert not restore_from_save or trainable  # don't restore un-trainable model
        assert not restore_from_save or save_dir is not None # can only restore if there is a save directory

        self.save_per_epoch = (save_dir is not None and trainable)
        self.shuffle = True
        self.restore_from_save = restore_from_save
        self.save_dir = save_dir
        self.trainable = trainable
        self.tensorboard_name = tensorboard_name
        self.load_scopes = []
        self.input_placeholders = {}
        self.output_tensors = {}
        self.train_ops = {}
        self.loss = None
        self._create_standard_placeholders()
        self.build()
        self._initialize_loss()
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = tf_log_level
        if self.tensorboard_name is not None:
            create_tensorboard_visualization(self.tensorboard_name)
        self.init = tf.global_variables_initializer()
        self.sess = tf.InteractiveSession()
        self.sess.run(self.init)
        if self.trainable:
            self.saver = tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=10)
        else:
            self.saver = None
        if self.restore_from_save:
            for each_scope in self.load_scopes:
                load_scope_from_save(self.save_dir, self.sess, each_scope)
            if len(self.load_scopes) == 0:
                restore_model_from_save(self.save_dir, self.sess)

    def _create_standard_placeholders(self):
        """"""
        self.input_placeholders['is training'] = tf.placeholder_with_default(False, (), name='is_training')

    def _fill_standard_placeholders(self, is_training):
        # Must at least return empty dictionary.
        return {'is training': is_training}

    def _initialize_loss(self):
        """If user specifies loss to train on (using self.loss), create an Adam optimizer to minimize that loss,
        and add the optimizer operation to dictionary of train ops. Initialize optional placeholder
        for learning rate and add it to input placeholders under name 'learning rate'. Add loss tensor
        to output tensor dictionary under name 'loss', so it can be evaluated during training."""
        if self.loss is not None:
            learning_rate = tf.placeholder_with_default(.001, shape=(), name='learning_rate')
            self.train_ops['loss'] = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
            self.input_placeholders['learning rate'] = learning_rate
            self.output_tensors['loss'] = self.loss

    @abstractmethod
    def build(self):
        """Implement Tensorflow model and specify model placeholders and output tensors you wish to evaluate
        using self.input_placeholders and self.output_tensors dictionaries. Specify each entry as a name:tensor pair.
        Specify variable scopes to restore by adding to self.load_scopes list. Specify loss function to train on
        by assigning loss tensor to self.loss variable. Read initialize_loss() documentation for adaptive
        learning rates and evaluating loss tensor at runtime."""
        pass

    def action_per_epoch(self, output_tensor_dict, epoch_index, is_training, **kwargs):
        """Optional: Define action to take place at the end of every epoch. Can use this
        for printing accuracy, saving statistics, etc. Remember, if is_training=False, we are using the model for
        prediction. Check for this. Returns true to continue training. Only return false if you wish to
        implement early-stopping."""
        return True

    def action_per_batch(self, input_batch_dict, output_batch_dict, epoch_index, batch_index, is_training, **kwargs):
        """Optional: Define action to take place at the end of every batch. Can use this
        for printing accuracy, saving statistics, etc. Remember, if is_training=False, we are using the model for
        prediction. Check for this."""
        pass

    def action_before_training(self, placeholder_dict, num_epochs, is_training, output_tensor_names=None, batch_size=32,
                               train_op_names=None, **kwargs):
        """Optional: Define action to take place at the beginning of training/prediction, once. This could be
        used to set output_tensor_names so that certain ops always execute, as needed for other action functions."""
        pass

    def _eval(self, dataset, num_epochs, parameter_dict=None, output_tensor_names=None, batch_size=32, is_training=True,
              train_op_names=None, **kwargs):
        """Evaluate output tensors of model with dataset as input. Optionally train on that dataset. Return dictionary
        of evaluated tensors to user. For internal use only, shared functionality between training and prediction."""

        # united for the cause of training!
        united_parameter_dict = self._fill_standard_placeholders(is_training)
        if parameter_dict is not None:
            united_parameter_dict.update(parameter_dict)

        # Allow dictionary as input!
        if isinstance(dataset, dict):
            dataset = DictionaryDataset(dataset)

        # Only train if model is trainable
        if is_training and not self.trainable:
            raise ValueError('Cannot train while model is not trainable.')

        self.action_before_training(dataset, num_epochs, is_training,
                                    output_tensor_names=output_tensor_names,
                                    batch_size=batch_size, train_op_names=train_op_names, **kwargs)

        # Control what train ops are executed via arguments
        if is_training:
            if train_op_names is None:
                train_op_list = [self.train_ops[op_name] for op_name in self.train_ops]
            else:
                train_op_list = [self.train_ops[op_name] for op_name in train_op_names]
        else:
            train_op_list = []

        # If user doesn't specify output tensors, evaluate them all!
        if output_tensor_names is None:
            output_tensor_names = [name for name in self.output_tensors]

        # Create list of output tensors, initialize output dictionaries
        output_tensors = [self.output_tensors[each_tensor_name] for each_tensor_name in output_tensor_names]
        all_output_batch_dicts = None

        print(united_parameter_dict)

        # Create feed dictionary for model parameters
        parameter_feed_dict = {self.input_placeholders[feature_name]: united_parameter_dict[feature_name]
                               for feature_name in united_parameter_dict}

        continue_training = True
        do_shuffle = self.shuffle and is_training

        # Evaluate and/or train on dataset. Run user-defined action functions
        for epoch_index in range(num_epochs):
            if not continue_training:
                break

            all_output_batch_dicts = []
            for batch_index, batch_dict in enumerate(dataset.generate_batches(batch_size=batch_size, shuffle=do_shuffle)):
                # Run batch in session - combine dataset features and parameters
                feed_dict = {self.input_placeholders[feature_name]: batch_dict[feature_name]
                             for feature_name in batch_dict}
                feed_dict.update(parameter_feed_dict)

                if not is_training:
                    assert len(train_op_list) == 0

                output_numpy_arrays, _ = self.sess.run([output_tensors, train_op_list], feed_dict)

                input_batch_dict = {feature_name: feed_dict[self.input_placeholders[feature_name]]
                                    for feature_name in batch_dict}
                output_batch_dict = {output_tensor_names[index]: output_numpy_arrays[index]
                                     for index in range(len(output_tensor_names))}

                # Keep history of batch outputs
                all_output_batch_dicts.append(output_batch_dict)

                self.action_per_batch(input_batch_dict, output_batch_dict, epoch_index,
                                      batch_index, is_training, **kwargs)

            if self.save_per_epoch and self.trainable and is_training:
                self.saver.save(self.sess, self.save_dir, global_step=epoch_index)

            # Call user action per epoch, and allow them to stop training early
            continue_training = self.action_per_epoch(all_output_batch_dicts, epoch_index, is_training, **kwargs)
            if not continue_training:
                break

        # Calculate output dictionary from last epoch executed
        output_dict_concat = concatenate_batch_dictionaries(all_output_batch_dicts)

        return output_dict_concat

    def train(self, dataset, output_tensor_names=None, num_epochs=5, parameter_dict=None, batch_size=32, **kwargs):
        """Train on a dataset. Can specify which output tensors to evaluate (or none at all if dataset is too large).
        Can specify batch size and provide parameters arguments as inputs to model placeholders. To add constant
        values for input placeholders, pass to parameter_dict a dictionary containing name:value pairs. Name must
        match internal name of desired placeholder as defined in self.input_placeholders dictionary. Can set number
        of epochs to train for. **kwargs can be used to provide additional arguments to internal action functions,
        which can be overloaded for extra functionality. Training examples are shuffled each epoch!

        Arguments:
            dataset - subclass object of Dataset class containing labelled input features. Can also be dictionary
            output_tensor_names - list of names of output tensors to evaluate. Names defined in build() function
            num_epochs - number of epochs to train on. Is possible to implement early stopping using action functions
            parameter_dict - dictionary of constant parameters to provide to model (like learning rates)
            batch_size - number of examples to train on at once
            kwargs - optional parameters sent to action functions for expanded functionality

        Returns: dictionary of evaluated output tensors.
        """
        output_tensor_dict = self._eval(dataset, num_epochs,
                                        parameter_dict=parameter_dict,
                                        output_tensor_names=output_tensor_names,
                                        batch_size=batch_size,
                                        train_op_names=None,
                                        is_training=True,
                                        **kwargs)

        return output_tensor_dict

    def predict(self, dataset, output_tensor_names=None, parameter_dict=None, batch_size=32, **kwargs):
        """Predict on a dataset. Can specify which output tensors to evaluate. Can specify batch size and provide
        parameters arguments as inputs to model placeholders. To add constant values for input placeholders, pass to
        parameter_dict a dictionary containing name:value pairs. Name must match internal name of desired placeholder
        as defined in self.input_placeholders dictionary. **kwargs can be used to provide additional arguments to
        internal action functions, which can be overloaded for extra functionality.

        Arguments:
            dataset - subclass object of Dataset class containing labelled input features. Can also be dictionary
            output_tensor_names - list of names of output tensors to evaluate. Names defined in build() function
            parameter_dict - dictionary of constant parameters to provide to model (like learning rates)
            batch_size - number of examples to train on at once
            kwargs - optional parameters sent to action functions for expanded functionality

        Returns: dictionary of evaluated output tensors."""
        output_tensor_dict = self._eval(dataset, 1,
                                        parameter_dict=parameter_dict,
                                        output_tensor_names=output_tensor_names,
                                        batch_size=batch_size,
                                        train_op_names=[],
                                        is_training=False,
                                        **kwargs)

        return output_tensor_dict


class Dataset(ABC):
    """Input to all GenericModel subclasses. Overload to implement a variety of datasets.
    For a dataset to apply, must be able to index into dataset to return dictionary of features
    for a single example. Must implement __len__ function to specify size of dataset. Complete
    processing of dataset is encouraged to occur within a Dataset object. If retrieving single
    examples during training is too slow, the generate_batches() function can be overridden."""
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
    def __init__(self, batch_feature_dict):
        """Create simple dataset from a dictionary of feature_name:numpy_feature entries.
        Interpret as a series of features, each having a name the model can refer to.
        Optionally, define dictionary of scalar features which remain the same across
        examples. The first dimension of all features must be equal to the length
        of the dataset."""
        self.batch_feature_dict = batch_feature_dict

        # Determine length of dataset
        self.length = -1
        for feature_name in self.batch_feature_dict:
            feature_length = self.batch_feature_dict[feature_name].shape[0]
            if self.length == -1:
                self.length = feature_length
            else:
                if self.length != feature_length:
                    raise ValueError('All batch features must have same length')

        if self.length <= 0:
            raise ValueError('Cannot have zero-length dataset.')

    def __getitem__(self, index):
        item_dict = {}
        for feature_name in self.batch_feature_dict:
            item_dict[feature_name] = self.batch_feature_dict[feature_name][index]
        return item_dict

    def __len__(self):
        return self.length

    def to_dict(self):
        return self.batch_feature_dict


class MNISTTrainSet:
    def __init__(self, num_examples=None):
        """Create dataset of MNIST examples for training on."""
        self.num_examples = num_examples

    def __getitem__(self):
        # Not used
        pass

    def __len__(self):
        # Not used
        pass

    def generate_batches(self, batch_size, shuffle=True):
        # Shuffle parameter not used
        mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
        if self.num_examples is None:
            self.num_examples = mnist.train.num_examples
        counter = 0

        while True:
            batch_xs, batch_ys = mnist.train.next_batch(batch_size, shuffle=shuffle)

            if batch_xs.shape[0] != 0 and (counter * batch_size < self.num_examples):
                yield {'image': batch_xs, 'label': batch_ys}
            else:
                break

            counter += 1


class MNISTTestSet(DictionaryDataset):
    """Create dataset of MNIST examples for testing.

    Features:
        'image': numpy array of 784 pixel images (flattened from 28 x 28)
        'label': numpy array of labels 0 - 9 indicating digit contained in each image"""

    def __init__(self):
        self.mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
        DictionaryDataset.__init__(self, {'image': self.mnist.test.images, 'label': self.mnist.test.labels})


def create_tensorboard_visualization(model_name):
    """Saves the Tensorflow graph of your model, so you can view it in a TensorBoard console."""
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
    """Concatenates numpy dictionaries. If numpy arrays represent single examples (no batch axis),
    set single_examples=True. Otherwise false.
    batch_dictionaries - list of dictionaries, all containing identical keys, each key being
    a feature name
    single_examples - decides whether to concatenate (for batches) or to stack (for single vectors)"""
    result = {}

    for key in batch_dictionaries[0]:
        if single_examples or len(batch_dictionaries[0][key].shape) == 0:
            result[key] = np.stack([d[key] for d in batch_dictionaries], axis=0)
        else:
            result[key] = np.concatenate([d[key] for d in batch_dictionaries], axis=0)

    return result


# GENERIC MODEL SUBCLASS EXAMPLES ######################################################################################


class SimpleModel(GenericModel):
    """Subclass of generic model that does not train, only performs static operation (adds 3 to input)."""
    def build(self):
        self.trainable = False
        tf_input = tf.placeholder(tf.float32, shape=(None, 1), name='x')
        tf_output = tf_input + 3.0
        self.input_placeholders['x'] = tf_input
        self.output_tensors['y'] = tf_output

    def action_per_epoch(self, output_tensor_dict, epoch_index, is_training, **kwargs):
        print('Executing action_per_epoch')

    def action_per_batch(self, input_batch_dict, output_batch_dict, epoch_index, batch_index, is_training, **kwargs):
        print('Executing action_per_batch')

    def action_before_training(self, placeholder_dict, num_epochs, is_training, output_tensor_names=None, batch_size=32,
                           train_op_names=None, save_per_epoch=True, **kwargs):
        return False  # Do not save after each epoch


class LessSimpleModel(GenericModel):
    """Subclass of generic model that learns a scalar value w based on training examples.
    Defines only the loss function, no optimizer. Learns to add a scalar to input."""
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

    def action_before_training(self, placeholder_dict, num_epochs, is_training, output_tensor_names=None,
                               batch_size=32, train_op_names=None, **kwargs):
        self.save_per_epoch = False


class EvenLessSimpleModel(GenericModel):
    """Subclass of generic model that learns a scalar value w based on training examples.
    User defines the Adam optimizer internally, making it even less simple. Learns to add
    a scalar to input."""
    def build(self):
        tf_input = tf.placeholder(tf.float32, shape=(None, 1), name='x')
        tf_w = tf.get_variable('w', (1, 1), initializer=tf.contrib.layers.xavier_initializer())
        tf_output = tf_input + tf_w
        self.input_placeholders['x'] = tf_input
        self.output_tensors['y'] = tf_output
        self.output_tensors['w'] = tf_w

        tf_label = tf.placeholder(tf.float32, shape=(None, 1), name='label')
        self.loss = tf.nn.l2_loss(tf_label - self.output_tensors['y'])
        self.input_placeholders['label'] = tf_label

    def action_before_training(self, placeholder_dict, num_epochs, is_training, output_tensor_names=None,
                               batch_size=32, train_op_names=None, **kwargs):
        self.save_per_epoch = False


# UNIT TESTS ###########################################################################################################


class GenericModelTest(unittest2.TestCase):
    def test_batch_generator_and_dictionary_dataset_arange(self):
        """non-shuffled batches correspond to input feature dictionary. Test batch size."""
        data = np.random.uniform(size=(50, 3))
        feature_dict = {'f1': data[:, 0], 'f2': data[:, 1], 'f3': data[:, 2]}
        dts = DictionaryDataset(feature_dict)
        batch_size = 10
        for index, batch_dict in enumerate(dts.generate_batches(batch_size=batch_size, shuffle=False)):
            for feature in batch_dict:
                #print('batch_dict: ' + str(batch_dict[feature]))
                #print('feature_dict: ' + str(feature_dict[feature][index*batch_size:index*batch_size+batch_size]))
                assert batch_dict[feature].shape[0] == batch_size
                assert np.array_equal(batch_dict[feature], feature_dict[feature][index*batch_size:index*batch_size+batch_size])

    def test_batch_generator_and_dictionary_dataset_shuffle(self):
        """Set shuffle to true and batches will not correspond to input feature dictionary"""
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
        """Test that DictionaryDataset batches correspond to input dataset."""
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
        """Test that batch features have correct shape and DD can handle remainders."""
        data = np.random.uniform(size=(10, 3))
        feature_dict = {'f1': data[:, 0], 'f2': data[:, 1], 'f3': data[:, 2]}
        dts = DictionaryDataset(feature_dict)
        batch_size = 20
        for batch_dict in dts.generate_batches(batch_size=batch_size, shuffle=False):
            assert batch_dict['f1'].shape == batch_dict['f2'].shape
            assert batch_dict['f2'].shape == batch_dict['f3'].shape
            assert batch_dict['f1'].shape[0] == 10

    def test_batch_generator_and_empty_dictionary_dataset(self):
        """You cannot create an empty dataset."""
        with self.assertRaises(ValueError):
            DictionaryDataset({})

    def test_dictionary_dataset(self):
        """Test that DictionaryDataset can handle multiple features."""
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
        """Test that the DictionaryDataset can handle multi-dimensional features."""
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
        """Test that SimpleModel can be instantiated."""
        SimpleModel('/tmp/sm_save/', 'sm')

    def test_simple_model_prediction(self):
        """Test that SimpleModel adds 3 to input."""
        sm = SimpleModel('/tmp/sm_save/', 'sm')

        dataset = DictionaryDataset({'x': np.array([[3], [4]])})

        output_dict = sm.predict(dataset)

        print(output_dict)

        assert np.array_equal(output_dict['y'], np.array([[6.], [7.]]))

    def test_simple_model_train(self):
        """SimpleModel should not be able to train. Confirm this."""
        sm = SimpleModel('/tmp/sm_save/', 'sm')

        d = DictionaryDataset({'x': np.array([[3], [4]])})

        with self.assertRaises(ValueError):
            sm.train(d, num_epochs=10)

    def test_less_simple_model_train(self):
        """Trains LessSimpleModel for 10000 epochs to converge w value to 3."""
        with tf.Graph().as_default():
            lsm = LessSimpleModel('/tmp/lsm_save/', 'lsm')

            dataset = DictionaryDataset({'x': np.array([[3]]), 'label': np.array([[6]])})

            output_dict = lsm.train(dataset, num_epochs=10000)
            epsilon = .01
            assert np.abs(3 - output_dict['w']) < epsilon
            print(output_dict)

            output_dict = lsm.predict(dataset, ['y'])

            print(output_dict['y'])

            assert np.abs(output_dict['y'] - 6) < epsilon

    def test_even_less_simple_model_train(self):
        """Train EvenLessSimpleModel for 10000 epochs to converge w value to 3."""
        with tf.Graph().as_default():
            lsm = EvenLessSimpleModel('/tmp/lsm_save/', 'lsm')

            dataset = DictionaryDataset({'x': np.array([[3]]), 'label': np.array([[6]])})

            output_dict = lsm.train(dataset, num_epochs=10000)
            epsilon = .1
            assert np.abs(3 - output_dict['w']) < epsilon
            assert output_dict['loss'] < epsilon
            assert len(lsm.train_ops) == 1
            print(output_dict)

            output_dict = lsm.predict(dataset, ['y', 'w'])

            assert np.abs(output_dict['y'] - 6) < epsilon

