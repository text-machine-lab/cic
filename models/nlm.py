"""Neural language model for generating sentences."""

from arcadian.gm import GenericModel
import numpy as np
import tensorflow as tf

class NeuralLanguageModelTraining(GenericModel):
    def __init__(self, max_len, vocab_len, emb_size, rnn_size, **kwargs):
        """Neural language model used to generate English sentences."""

        self.max_len = max_len  # maximum sentence length
        self.vocab_len = vocab_len  # number of vocabulary words
        self.emb_size = emb_size  # size of each word embedding
        self.rnn_size = rnn_size  # size of the LSTM cell

        super().__init__(**kwargs)

    def build(self):
        """Implement Tensorflow model and specify model placeholders and output tensors you wish to evaluate
        using self.input_placeholders and self.output_tensors dictionaries. Specify each entry as a name:tensor pair.
        Specify variable scopes to restore by adding to self.load_scopes list. Specify loss function to train on
        by assigning loss tensor to self.loss variable. Read initialize_loss() documentation for adaptive
        learning rates and evaluating loss tensor at runtime."""

        # Specify inputs

        with tf.variable_scope('LANGUAGE_MODEL'):
            self.tf_messages = tf.placeholder(dtype=tf.int32, shape=(None, self.max_len)) # label
            batch_size = tf.shape(self.tf_messages)[0]

            # Build embedding matrix
            self.tf_learned_embeddings = tf.get_variable('embs', (self.vocab_len, self.emb_size),
                                                       initializer=tf.random_normal_initializer())

            self.tf_message_embs = tf.nn.embedding_lookup(self.tf_learned_embeddings, self.tf_messages)

            tf_message_prediction, tf_message_logits, tf_message_prob \
                = self.build_decoder(batch_size)

        self.build_trainer(tf_message_logits, self.tf_messages)

        self.i.update({'message': self.tf_messages})

        self.o.update({'prediction': tf_message_prediction,
                             'probabilities': tf_message_prob})

        self.load_scopes = ['LANGUAGE_MODEL']

    def build_decoder(self, batch_size):
        """Build decoder portion of autoencoder in Tensorflow."""
        # USE_TEACHER_FORCING boolean is not yet implemented!!!
        m = batch_size
        with tf.variable_scope('MESSAGE_DECODER'):
            # tf_decoder_input_tile = tf.tile(tf.reshape(tf_decoder_input, [-1, 1, self.rnn_size]),
            #                                 [1, self.max_message_size, 1])
            output_weight = tf.get_variable('output_weight',
                                            shape=[self.rnn_size, self.emb_size],
                                            initializer=tf.contrib.layers.xavier_initializer())
            output_bias = tf.get_variable('output_bias',
                                          shape=[self.emb_size],
                                          initializer=tf.contrib.layers.xavier_initializer())

            tf_go_token = tf.get_variable('go_token', shape=[1, self.emb_size])
            tf_go_token_tile = tf.tile(tf_go_token, [m, 1])

            response_lstm = tf.contrib.rnn.LSTMCell(num_units=self.rnn_size, state_is_tuple=False)
            tf_hidden_state = response_lstm.zero_state(m, tf.float32)
            all_word_logits = []
            all_word_probs = []
            all_word_predictions = []
            for i in range(self.max_len):
                if i == 0:
                    tf_teacher_signal = tf_go_token_tile  # give model go token
                else:
                    tf_teacher_signal = self.tf_message_embs[:, i - 1, :]  # @i=1, selects first label word

                tf_decoder_input = tf_teacher_signal

                tf_output, tf_hidden_state = response_lstm(tf_decoder_input, tf_hidden_state)
                tf_word_emb = tf.tanh(tf.matmul(tf_output, output_weight) + output_bias)
                tf_word_logits = tf.matmul(tf_word_emb, self.tf_learned_embeddings, transpose_b=True)
                tf_word_prob = tf.nn.softmax(tf_word_logits)
                tf_word_prediction = tf.argmax(tf_word_logits, axis=1)

                all_word_logits.append(tf_word_logits)
                all_word_probs.append(tf_word_prob)
                all_word_predictions.append(tf_word_prediction)

            with tf.name_scope('decoder_outputs'):
                tf_message_logits = tf.stack(all_word_logits, axis=1)
                tf_message_prob = tf.stack(all_word_probs, axis=1)
                tf_message_prediction = tf.stack(all_word_predictions, axis=1)

        return tf_message_prediction, tf_message_logits, tf_message_prob

    def build_trainer(self, tf_message_log_prob, tf_message):
        """Calculate loss function and construct optimizer op
        for 'tf_message_log_prob' prediction and 'tf_message' label."""
        with tf.variable_scope('LOSS'):
            tf_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=tf_message_log_prob,
                                                                       labels=tf_message,
                                                                       name='word_losses')
            self.loss = tf.reduce_mean(tf_losses)

        return self.loss

    def action_per_batch(self, input_batch_dict, output_batch_dict, epoch_index, batch_index, is_training,
                         params, **kwargs):

        # Save every 1000 batches!
        if batch_index != 0 and batch_index % 1000 == 0 and is_training:
            print('Saving...')
            if self.save_per_epoch and self.trainable and is_training:
                self.saver.save(self.sess, self.save_dir, global_step=epoch_index)

    def action_per_epoch(self, output_tensor_dict, epoch_index, is_training, params, **kwargs):

        print('Loss: %s' % np.mean(output_tensor_dict['loss']))

        return True


class NeuralLanguageModelPrediction(GenericModel):
    def __init__(self, vocab_len, emb_size, rnn_size, **kwargs):
        """Neural language model used to generate English sentences."""

        self.vocab_len = vocab_len  # number of vocabulary words
        self.emb_size = emb_size  # size of each word embedding
        self.rnn_size = rnn_size  # size of the LSTM cell

        super().__init__(**kwargs)

    def build(self):
        """Implement Tensorflow model and specify model placeholders and output tensors you wish to evaluate
        using self.input_placeholders and self.output_tensors dictionaries. Specify each entry as a name:tensor pair.
        Specify variable scopes to restore by adding to self.load_scopes list. Specify loss function to train on
        by assigning loss tensor to self.loss variable. Read initialize_loss() documentation for adaptive
        learning rates and evaluating loss tensor at runtime."""

        with tf.variable_scope('LANGUAGE_MODEL'):

            # We want to insert the chosen word embedding into the model
            self.tf_teacher_signal = tf.placeholder(tf.float32, shape=(None, self.emb_size), name='teacher_signal')

            batch_size = tf.shape(self.tf_teacher_signal)[0]

            # We want to be able to insert a word and evaluate its embedding
            self.tf_word = tf.placeholder(tf.int32, shape=(None,), name='word')

            with tf.variable_scope('MESSAGE_DECODER'):
                self.response_lstm = tf.contrib.rnn.LSTMCell(num_units=self.rnn_size, state_is_tuple=False)

                # We want to initially evaluate the go token
                self.tf_go_token = tf.get_variable('go_token', shape=[1, self.emb_size])

                # We want to insert the previous hidden state into the model
                self.tf_hidden = tf.placeholder(tf.float32, shape=(None, self.rnn_size * 2), name='hidden')

            # We want to be able to evaluate one copy of the initial hidden state
            self.tf_init_hidden = self.response_lstm.zero_state(1, tf.float32)

            # Build embedding matrix
            self.tf_learned_embeddings = tf.get_variable('embs', (self.vocab_len, self.emb_size),
                                                         initializer=tf.random_normal_initializer())

            self.tf_word_emb = tf.nn.embedding_lookup(self.tf_learned_embeddings, self.tf_word)

            # We want to be able to run the LSTM on a teacher signal embedding and a previous hidden
            # state, and evaluate word probabilities.
            tf_word_prediction, tf_word_logits, tf_word_prob, tf_new_hidden_state \
                = self.build_lstm(self.tf_teacher_signal, self.tf_hidden)

        # Set interface
        self.load_scopes = ['LANGUAGE_MODEL']

        self.i.update({'teacher_signal': self.tf_teacher_signal,
                            'hidden': self.tf_hidden,
                            'word': self.tf_word})

        self.o.update({'prediction': tf_word_prediction,
                             'probabilities': tf_word_prob,
                             'go_token': self.tf_go_token,
                             'word_emb': self.tf_word_emb,
                             'init_hidden': self.tf_init_hidden,
                             'hidden': tf_new_hidden_state,
                             'embs': self.tf_learned_embeddings})

    def build_lstm(self, tf_teacher_signal, hidden_state):
        """Build decoder portion of autoencoder in Tensorflow."""
        # USE_TEACHER_FORCING boolean is not yet implemented!!!
        with tf.variable_scope('MESSAGE_DECODER'):
            # tf_decoder_input_tile = tf.tile(tf.reshape(tf_decoder_input, [-1, 1, self.rnn_size]),
            #                                 [1, self.max_message_size, 1])
            output_weight = tf.get_variable('output_weight',
                                            shape=[self.rnn_size, self.emb_size],
                                            initializer=tf.contrib.layers.xavier_initializer())
            output_bias = tf.get_variable('output_bias',
                                          shape=[self.emb_size],
                                          initializer=tf.contrib.layers.xavier_initializer())

            tf_output, tf_hidden_state = self.response_lstm(tf_teacher_signal, hidden_state)
            tf_word_emb = tf.tanh(tf.matmul(tf_output, output_weight) + output_bias)
            tf_word_logits = tf.matmul(tf_word_emb, self.tf_learned_embeddings, transpose_b=True)
            tf_word_prob = tf.nn.softmax(tf_word_logits)
            tf_word_prediction = tf.argmax(tf_word_logits, axis=1)


        return tf_word_prediction, tf_word_logits, tf_word_prob, tf_hidden_state

