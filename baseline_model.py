"""Copyright 2017 David Donahue. Baseline LSTM model to make predictions on SQuAD dataset."""
import unittest2
import tensorflow as tf


class LSTMBaselineModel:
    def __init__(self):
        pass

    def build(self):
        pass

    def __call__(self, features):
        pass


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


def build_gru(gru_hidden_dim, tf_batch_size, inputs, num_time_steps, gru_scope=None,
              reuse=False, time_step_inputs=None, reverse=False):
    """Runs an LSTM over input data and returns LSTM output and hidden state. Arguments:
    lstm_hidden_dim - Size of hidden state of LSTM
    tf_batch_size - Tensor value representing size of current batch. Required for LSTM package
    inputs - Full input into LSTM. List of tensors as input. Per tensor: First dimension of m examples, with second dimension holding concatenated input for all timesteps
    input_time_step_size - Size of input from tf_input that will go into LSTM in a single timestep
    num_time_steps - Number of time steps to run LSTM
    lstm_scope - Can be a string or a scope object. Used to disambiguate variable scopes of different LSTM objects
    time_step_inputs - Inputs that are per time step. The same tensor is inserted into the model at each time step
    reverse - flag indicating whether the inputs should be fed in reverse order. useful for bidirectional GRU

    Returns: list of num_time_step GRU outputs and list of num_time_step GRU hidden states."""
    if time_step_inputs is None:
        time_step_inputs = []
    time_step_outputs = []
    time_step_hidden_states = []
    gru = tf.contrib.rnn.GRUCell(num_units=gru_hidden_dim)
    #gru = tf.contrib.rnn.AttentionCellWrapper(gru, gru_hidden_dim)
    tf_hidden_state = gru.zero_state(tf_batch_size, tf.float32)
    for i in range(num_time_steps):
        # Grab time step input for each input tensor
        current_time_step_inputs = []
        for tf_input in inputs:
            if not reverse:
                current_time_step_inputs.append(tf_input[:, i, :])
            else:
                current_time_step_inputs.append(tf_input[:, num_time_steps - i - 1, :])
        #tf.slice(tf_input, [0, i, 0], [-1, i, input_time_step_size]))

        tf_input_time_step = tf.concat(current_time_step_inputs + time_step_inputs, 1)

        with tf.variable_scope(gru_scope) as scope:
            if i > 0 or reuse:
                scope.reuse_variables()
            tf_lstm_output, tf_hidden_state = gru(tf_input_time_step, tf_hidden_state)
            # Return outputs at all timesteps to caller
            if i == 0:
                print('tf_lstm_output shape: %s' % str(tf_lstm_output.get_shape()))
                print('tf_hidden_state shape: %s' % str(tf_hidden_state.get_shape()))
            time_step_outputs.append(tf_lstm_output)
            time_step_hidden_states.append(tf_hidden_state)

    return time_step_outputs, time_step_hidden_states


def create_dense_layer(input_layer, input_size, output_size, activation=None, include_bias=True, name=None):
    with tf.name_scope(name):
        tf_w = tf.Variable(tf.random_normal([input_size, output_size], stddev=.1))
        tf_b = tf.Variable(tf.random_normal([output_size]))
        output_layer = tf.matmul(input_layer, tf_w)
        if include_bias:
            output_layer = output_layer + tf_b
        if activation == 'relu':
            output_layer = tf.nn.relu(output_layer)
        elif activation == 'sigmoid':
            output_layer = tf.nn.sigmoid(output_layer)
        elif activation is None:
            pass
        else:
            print('Error: Did not specify layer activation')

    # regularizer = slim.l2_regularizer(reg_const)
    # regularizer_loss = regularizer(tf_w) + regularizer(tf_b)
    # slim.losses.add_loss(regularizer_loss)

    return output_layer, tf_w, tf_b


class LSTMBaselineModelTest(unittest2.TestCase):
    def setUp(self):
        pass

