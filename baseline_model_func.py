"""Copyright 2017 David Donahue. Functions and unit tests for baseline model script."""
import unittest2
import tensorflow as tf
import config
import numpy as np
import squad_dataset_tools as sdt


class LSTMBaselineModel:
    def __init__(self, np_embedding, rnn_size, learning_rate, save_dir=None, restore_from_save=False):
        self.np_embedding = np_embedding
        self.rnn_size = rnn_size

        self.tf_question_indices, self.tf_context_indices, self.tf_answer_indices, self.tf_answer_masks, self.tf_batch_size, \
            self.tf_keep_prob, self.tf_predictions, self.tf_probabilities, self.tf_log_probabilities, self.tf_question_embs, self.tf_context_embs \
            = self.build(self.np_embedding, self.rnn_size)

        self.tf_total_loss = self.build_trainer(self.tf_log_probabilities, self.tf_answer_indices, self.tf_answer_masks)

        self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.tf_total_loss)
        self.init = tf.global_variables_initializer()
        self.sess = tf.InteractiveSession()
        with tf.name_scope("SAVER"):
            self.saver = tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=10)
        self.sess.run(self.init)

        if save_dir is not None and restore_from_save:
            print('Restoring from save...')
            restore_model_from_save(save_dir, var_list=tf.trainable_variables(), sess=self.sess)

        self.model_io = {'questions': self.tf_question_indices,
                    'answers': self.tf_answer_indices,
                    'contexts': self.tf_context_indices,
                    'probabilities': self.tf_probabilities,
                    'predictions': self.tf_predictions,
                    'batch_size': self.tf_batch_size}

    def build(self, np_embeddings, rnn_size):
        print('Loading embeddings into Tensorflow')
        tf_embeddings = tf.Variable(np_embeddings, name='word_embeddings', dtype=tf.float32, trainable=False)
        print('Constructing placeholders')

        with tf.name_scope('PLACEHOLDERS'):
            tf_question_indices = tf.placeholder(dtype=tf.int32, shape=(None, config.MAX_QUESTION_WORDS),
                                                 name='question_indices')
            # tf_question_lengths = tf.placeholder(dtype=tf.int32, shape=(None), name='question_lengths')
            tf_context_indices = tf.placeholder(dtype=tf.int32, shape=(None, config.MAX_CONTEXT_WORDS),
                                                name='context_indices')
            # tf_context_lengths = tf.placeholder(dtype=tf.int32, shape=(None), name='context_lengths')
            tf_answer_indices = tf.placeholder(dtype=tf.int32, shape=(None, config.MAX_ANSWER_WORDS),
                                               name='answer_indices')
            tf_answer_masks = tf.placeholder(dtype=tf.float32, shape=(None, config.MAX_ANSWER_WORDS),
                                             name='answer_masks')
            tf_batch_size = tf.placeholder(dtype=tf.int32, shape=(), name='batch_size')
            tf_keep_prob = tf.placeholder_with_default(1.0, shape=(), name='keep_prob')

        with tf.name_scope('INPUT_EMBEDDINGS'):
            tf_question_embs = tf.nn.embedding_lookup(tf_embeddings, tf_question_indices, name='question_embeddings')
            tf_question_embs_dropout = tf.nn.dropout(tf_question_embs, tf_keep_prob)
            tf_context_embs = tf.nn.embedding_lookup(tf_embeddings, tf_context_indices, name='context_embeddings')
            tf_context_embs_dropout = tf.nn.dropout(tf_context_embs, tf_keep_prob)
        # Correct so far...

        print('Question embeddings shape: %s' % str(tf_question_embs.shape))
        print('Context embeddings shape: %s' % str(tf_context_embs.shape))

        # Removed sequence lengths from question and context encoders

        # Model
        with tf.variable_scope('QUESTION_ENCODER'):
            question_lstm = tf.contrib.rnn.LSTMCell(num_units=rnn_size)
            tf_question_outputs, tf_question_state = tf.nn.dynamic_rnn(question_lstm, tf_question_embs_dropout,
                                                                       sequence_length=None, dtype=tf.float32)

        # tf_question_state_reshape = tf.reshape(tf_question_state, [-1, 1, RNN_HIDDEN_DIM])
        # tf_question_state_tile = tf.tile(tf_question_state_reshape, [1, config.MAX_CONTEXT_WORDS, 1])
        # tf_context_encoder_input = tf.concat([tf_context_embs, tf_question_state_tile], axis=2)
        # assert tf_context_encoder_input.shape[1].value == config.MAX_CONTEXT_WORDS
        # assert tf_context_encoder_input.shape[2].value == RNN_HIDDEN_DIM + config.GLOVE_EMB_SIZE

        with tf.variable_scope('CONTEXT_ENCODER'):
            context_lstm = tf.contrib.rnn.LSTMCell(num_units=rnn_size)
            tf_context_outputs, tf_context_state = tf.nn.dynamic_rnn(context_lstm, tf_context_embs_dropout,
                                                                     sequence_length=None, dtype=tf.float32)

        with tf.variable_scope('MATCH_GRU'):
            with tf.variable_scope('FORWARD'):
                Hr_forward = match_gru(tf_question_outputs, tf_context_outputs, tf_batch_size, rnn_size)
            with tf.variable_scope('BACKWARD'):
                Hr_backward = match_gru(tf_question_outputs, tf.reverse(tf_context_outputs, [1]), tf_batch_size, rnn_size)
            Hr = tf.concat([Hr_forward, tf.reverse(Hr_backward, [1])], axis=2)
            Hr_dropout = tf.nn.dropout(Hr, tf_keep_prob)

            Hr_tilda = tf.concat([tf.zeros([tf_batch_size, 1, rnn_size * 2]), Hr_dropout], axis=1,
                                 name='Hr_tilda')

        with tf.name_scope('OUTPUT'):
            tf_log_probabilities, all_hidden_states = pointer_net(Hr_tilda, tf_batch_size, rnn_size)

            tf_probabilities = tf.nn.softmax(tf_log_probabilities)

            tf_predictions = tf.argmax(tf_probabilities, axis=2, name='predictions')

        return tf_question_indices, tf_context_indices, tf_answer_indices, tf_answer_masks, tf_batch_size, \
               tf_keep_prob, tf_predictions, tf_probabilities, tf_log_probabilities, tf_question_embs, tf_context_embs

    def build_trainer(self, tf_log_probabilities, tf_answer_indices, tf_answer_masks):
        # Calculate loss per each
        with tf.name_scope('LOSS'):
            time_step_losses = []
            for time_step in range(config.MAX_ANSWER_WORDS):
                time_step_losses.append(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(logits=tf_log_probabilities[:, time_step, :],
                                                                   labels=tf_answer_indices[:, time_step]))
            tf_total_losses = tf.stack(time_step_losses, axis=1)

            # tf_total_similarity_loss = 0
            # for i in range(config.MAX_ANSWER_WORDS - 1):
            #     tf_pair_similarity_loss = tf.reduce_sum(tf.multiply(tf_probabilities[:, i, :], tf_probabilities[:, i + 1, :]), axis=1)
            #     tf_total_similarity_loss += tf.reduce_mean(tf_pair_similarity_loss)

            tf_masked_losses = tf.multiply(tf_total_losses, tf_answer_masks)
            tf_total_loss = tf.reduce_mean(tf_masked_losses)
            # tf_total_loss += tf_total_similarity_loss  * SIMILARITY_LOSS_CONST
            # tf_total_loss = tf.reduce_mean(tf_total_losses) + tf_total_similarity_loss * SIMILARITY_LOSS_CONST

        return tf_total_loss

    def __call__(self, features, num_batches):
        pass

    def train(self, np_questions, np_contexts, np_answers, np_answer_masks, batch_size, num_epochs, keep_prob, print_per_n_batches=20):
        print('Training model...')
        all_train_predictions = []
        for epoch in range(num_epochs):
            print('Epoch: %s' % epoch)
            losses = []
            accuracies = []
            word_accuracies = []
            frac_zeros = []
            for i in range(num_epochs):
                np_question_batch = np_questions[i * batch_size:i * batch_size + batch_size, :]
                np_answer_batch = np_answers[i * batch_size:i * batch_size + batch_size, :]
                np_answer_mask_batch = np_answer_masks[i * batch_size:i * batch_size + batch_size, :]
                np_context_batch = np_contexts[i * batch_size:i * batch_size + batch_size, :]
                np_batch_predictions, np_loss, _ = self.sess.run([self.tf_predictions, self.tf_total_loss, self.train_op],
                                                            feed_dict={self.tf_question_indices: np_question_batch,
                                                                       self.tf_answer_indices: np_answer_batch,
                                                                       self.tf_answer_masks: np_answer_mask_batch,
                                                                       self.tf_context_indices: np_context_batch,
                                                                       self.tf_batch_size: batch_size,
                                                                       self.tf_keep_prob: keep_prob})
                accuracy, word_accuracy = sdt.compute_mask_accuracy(np_answer_batch,
                                                                    np_batch_predictions,
                                                                    np_answer_mask_batch)
                frac_zero = sdt.compute_multi_label_accuracy(np_batch_predictions,
                                                             np.zeros([batch_size, config.MAX_ANSWER_WORDS]))
                accuracies.append(accuracy)
                word_accuracies.append(word_accuracy)
                frac_zeros.append(frac_zero)
                if print_per_n_batches is not None and i % print_per_n_batches == 0:
                    print('Batch TRAIN EM Score: %s' % np.mean(accuracies))
                losses.append(np_loss)
                if epoch == num_epochs - 1:  # last epoch
                    all_train_predictions.append(np_batch_predictions)
            epoch_loss = np.mean(losses)
            epoch_accuracy = np.mean(accuracies)
            epoch_word_accuracy = np.mean(word_accuracies)
            epoch_frac_zero = np.mean(frac_zeros)
            print('Epoch loss: %s' % epoch_loss)
            print('Epoch TRAIN EM Accuracy: %s' % epoch_accuracy)
            print('Epoch TRAIN Word Accuracy: %s' % epoch_word_accuracy)
            print('Epoch fraction of zero vector answers: %s' % epoch_frac_zero)
            print('Saving model %s' % epoch)
            self.saver.save(self.sess, config.BASELINE_MODEL_SAVE_DIR, global_step=epoch)

        np_train_predictions = np.concatenate(all_train_predictions, axis=0)

        return np_train_predictions

    def predict_on_examples(self,
                            np_questions,
                            np_contexts,
                            batch_size):
        # Must generate validation predictions in batches to avoid OOM error
        assert np_questions.shape[0] == np_contexts.shape[0]
        num_val_examples = np_questions.shape[0]
        num_val_batches = int(num_val_examples / batch_size) + 1  # +1 to include remainder examples
        all_val_predictions = []
        all_val_probabilities = []
        for batch_index in range(num_val_batches):
            current_start_index = batch_size * batch_index
            if current_start_index + batch_size >= num_val_examples:
                effective_batch_size = num_val_examples - current_start_index
            else:
                effective_batch_size = batch_size
            if effective_batch_size == 0:
                break
            current_end_index = current_start_index + effective_batch_size
            np_batch_val_predictions, np_batch_val_probabilities = \
                tf.get_default_session().run([self.model_io['predictions'], self.model_io['probabilities']],
                                             feed_dict={self.model_io['questions']: np_questions[
                                                                               current_start_index:current_end_index,
                                                                               :],
                                                        self.model_io['contexts']: np_contexts[
                                                                              current_start_index:current_end_index, :],
                                                        self.model_io['batch_size']: effective_batch_size})
            all_val_predictions.append(np_batch_val_predictions)
            all_val_probabilities.append(np_batch_val_probabilities)
        np_val_predictions = np.concatenate(all_val_predictions, axis=0)
        np_val_probabilities = np.concatenate(all_val_probabilities, axis=0)

        return np_val_predictions, np_val_probabilities


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


def match_gru(tf_question_outputs, tf_passage_outputs, batch_size, hidden_size):
    """Match-LSTM implementation based on https://arxiv.org/pdf/1608.07905.pdf"""
    match_gru = tf.contrib.rnn.LSTMCell(num_units=hidden_size)
    tf_hidden_state = match_gru.zero_state(batch_size, tf.float32)
    tf_lstm_output = tf.zeros([batch_size, hidden_size])

    tf_question_weight = tf.get_variable('match_W_q', shape=[hidden_size, hidden_size], initializer=tf.contrib.layers.xavier_initializer())
    tf_passage_weight = tf.get_variable('match_W_p', shape=[hidden_size, hidden_size], initializer=tf.contrib.layers.xavier_initializer())
    tf_hidden_weight = tf.get_variable('match_W_r', shape=[hidden_size, hidden_size], initializer=tf.contrib.layers.xavier_initializer())
    tf_passage_bias = tf.get_variable('match_b_p', shape=[hidden_size], initializer=tf.contrib.layers.xavier_initializer())
    tf_attention_weight = tf.get_variable('match_w', shape=[hidden_size, 1], initializer=tf.contrib.layers.xavier_initializer())
    tf_attention_bias = tf.get_variable('match_b', shape=[1, 1], initializer=tf.contrib.layers.xavier_initializer())
    Hr_states = []
    for i in range(config.MAX_CONTEXT_WORDS):  # Could have a problem here...
        with tf.name_scope('MATCH_TIMESTEP'):
            hiP = tf_passage_outputs[:, i, :]
            Hq_reshape = tf.reshape(tf_question_outputs, [-1, hidden_size], name='question_reshape')
            Wq_Hq_matmul = tf.matmul(Hq_reshape, tf_question_weight, name='q_o_W_q_not_reshaped')
            Wq_Hq_matmul_reshape = tf.reshape(Wq_Hq_matmul, [batch_size, config.MAX_QUESTION_WORDS, hidden_size], name='q_o_W_q')
            Wp_hiP_matmul = tf.matmul(hiP, tf_passage_weight, name='context_att_emb')
            Wr_hr = tf.matmul(tf_lstm_output, tf_hidden_weight, name='hidden_att_emb')
            WphiP_Wrhr_bp_add = tf.reshape(Wp_hiP_matmul + Wr_hr + tf_passage_bias, [batch_size, 1, hidden_size], name='qcr_transform_reshaped')
            G_i = tf.tanh(Wq_Hq_matmul_reshape + WphiP_Wrhr_bp_add, name='G_i')
            a_i = tf.reshape(tf.matmul(tf.reshape(G_i, [-1, hidden_size]), tf_attention_weight) + tf_attention_bias, [batch_size, config.MAX_QUESTION_WORDS, 1], name='a_i')
            H_q_a_i = tf.reshape(tf.matmul(tf_question_outputs, tf.nn.softmax(a_i), transpose_a=True), [-1, hidden_size], name='H_q_a_i')
            tf_match_input = tf.concat([hiP, H_q_a_i], axis=1, name='match_input')
            with tf.variable_scope('MATCH_ENCODER') as scope:
                if i > 0:
                    scope.reuse_variables()
                tf_lstm_output, tf_hidden_state = match_gru(tf_match_input, tf_hidden_state)
                Hr_states.append(tf_lstm_output)
    return tf.concat([tf.reshape(state, [-1, 1, hidden_size]) for state in Hr_states], axis=1)


def pointer_net(Hr_tilda, batch_size, hidden_size):
    """Pointer Net implementation based on https://arxiv.org/pdf/1608.07905.pdf"""
    with tf.name_scope('POINTER_VARIABLES'):
        V = tf.get_variable('V', shape=[hidden_size * 2, hidden_size],
                            initializer=tf.contrib.layers.xavier_initializer())
        W_a = tf.get_variable('W_a', shape=[hidden_size, hidden_size],
                              initializer=tf.contrib.layers.xavier_initializer())
        b_a = tf.get_variable('b_a', shape=[hidden_size], initializer=tf.contrib.layers.xavier_initializer())
        v = tf.get_variable('w', shape=[hidden_size, 1], initializer=tf.contrib.layers.xavier_initializer())
        c = tf.get_variable('c', shape=[1, 1, 1], initializer=tf.contrib.layers.xavier_initializer())

    with tf.variable_scope('POINTER_NET') as scope:
        answer_lstm = tf.contrib.rnn.LSTMCell(num_units=hidden_size)
        tf_hidden_state = answer_lstm.zero_state(batch_size, tf.float32)
        tf_lstm_output = tf.zeros([batch_size, hidden_size])
        B_k_predictions = []
        all_hidden_states = []

        num_time_steps = Hr_tilda.shape[1].value
        assert num_time_steps == config.MAX_CONTEXT_WORDS + 1

        for i in range(config.MAX_ANSWER_WORDS):
            all_hidden_states.append(tf_hidden_state)
            with tf.name_scope('ANSWER_TIMESTEP'):
                Hr_tilda_V_matmul = tf.reshape(tf.matmul(tf.reshape(Hr_tilda, [-1, hidden_size * 2]), V),
                                               [-1, num_time_steps, hidden_size], name='Hr_tilda_V_matmul')
                F_k = tf.tanh(Hr_tilda_V_matmul + tf.reshape(tf.matmul(tf_lstm_output, W_a) + b_a,
                                                             [-1, 1, hidden_size]), name='F_k')  # Should broadcast
                F_k_v_matmul = tf.reshape(tf.matmul(tf.reshape(F_k, [-1, hidden_size]), v),
                                          [-1, num_time_steps, 1],
                                          name='F_k_v_matmul')
                B_k = tf.add(F_k_v_matmul, c, name='B_k')
                B_k_predictions.append(tf.reshape(B_k, [-1, 1, num_time_steps], name='B_k_reshape'))

                tf_answer_input = tf.reshape(tf.matmul(Hr_tilda, tf.nn.softmax(B_k), transpose_a=True),
                                             [-1, hidden_size * 2],
                                             name='answer_input')
                if i > 0:
                    scope.reuse_variables()
                tf_lstm_output, tf_hidden_state = answer_lstm(tf_answer_input, tf_hidden_state)
    tf_probabilities = tf.concat(B_k_predictions, axis=1, name='probabilities')
    return tf_probabilities, all_hidden_states


def create_dense_layer(input_layer, input_size, output_size, activation=None, include_bias=True, name=None, std=.1):
    with tf.name_scope(name):
        tf_w = tf.Variable(tf.random_normal([input_size, output_size], stddev=std))
        tf_b = tf.Variable(tf.zeros([output_size]))
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

