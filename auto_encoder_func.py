"""Helper classes and functions for auto_encoder_model.py"""
import tensorflow as tf
import baseline_model_func
import chat_model_func
import numpy as np
import time

MAX_MESSAGE_LENGTH = 10
MAX_NUMBER_OF_MESSAGES = None
STOP_TOKEN = '<STOP>'
DELIMITER = ' +++$+++ '
RNN_HIDDEN_DIM = 500
LEARNED_EMBEDDING_SIZE = 200
LEARNING_RATE = .0005
KEEP_PROB = .9
RESTORE_FROM_SAVE = False
BATCH_SIZE = 20
TRAINING_FRACTION = 0.9
NUM_EPOCHS = 1
NUM_EXAMPLES_TO_PRINT = 20
VALIDATE_ENCODER_AND_DECODER = False
SAVE_TENSORBOARD_VISUALIZATION = False
SHUFFLE_EXAMPLES = True
USE_REDDIT_MESSAGES = False
VARIATIONAL = True
REVERSE_INPUT_MESSAGES = True
SEED = 'hello world'


def convert_string_to_numpy(msg, nlp, vocab_dict):
    tk_message = nlp.tokenizer(msg.lower())
    tk_tokens = [str(token) for token in tk_message if str(token) != ' ' and str(token) in vocab_dict] + [STOP_TOKEN]
    np_message = chat_model_func.construct_numpy_from_messages([tk_tokens], vocab_dict, MAX_MESSAGE_LENGTH)
    return np_message


class AutoEncoder:
    """Auto-encoder model built in Tensorflow. Encodes English sentences as points
    in n-dimensional space(as codes) using an encoder RNN, then converts from that space
    back to the original sentence using a decoder RNN. Can be used on arbitrary input
    sequences other than English sentences. Represents input tokens as indices and
    learns an embedding per index."""
    def __init__(self, word_embedding_size, vocab_size, rnn_size, max_message_size,
                 encoder=True, decoder=True, learning_rate=None, save_dir=None, load_from_save=False,
                 variational=True, use_teacher_forcing=True):
        self.word_embedding_size = word_embedding_size
        self.encoder = encoder
        self.decoder = decoder
        self.rnn_size = rnn_size
        self.max_message_size = max_message_size
        self.vocab_size = vocab_size
        self.learning_rate = learning_rate
        self.save_dir = save_dir
        self.use_teacher_forcing = use_teacher_forcing

        assert encoder or decoder

        self.tf_message = tf.placeholder_with_default(tf.zeros([1, self.max_message_size], dtype=tf.int32), [None, self.max_message_size], name='input_message')
        self.tf_latent = tf.placeholder(dtype=tf.float32, shape=[None, self.rnn_size], name='latent_embedding')
        self.tf_keep_prob = tf.placeholder_with_default(1.0, (), name='keep_prob')
        self.tf_kl_const = tf.placeholder_with_default(1.0, (), name='kl_const')
        self.tf_is_training = tf.placeholder_with_default(False, (), name='model_is_training')
        with tf.variable_scope('LEARNED_EMBEDDINGS'):
            self.tf_learned_embeddings = tf.get_variable('learned_embeddings',
                                                         shape=[self.vocab_size, self.word_embedding_size],
                                                         initializer=tf.contrib.layers.xavier_initializer())
            self.tf_message_embs = tf.nn.embedding_lookup(self.tf_learned_embeddings, self.tf_message, name='message_embeddings')
        if self.encoder:
            self.tf_latent_message, self.tf_latent_mean, self.tf_latent_log_std \
                = self.build_encoder(self.tf_message_embs, self.tf_keep_prob, include_epsilon=(self.decoder and variational))
        if self.decoder:
            if self.encoder:
                decoder_input = self.tf_latent_message
            else:
                decoder_input = self.tf_latent
            self.tf_message_prediction, self.tf_message_log_prob, self.tf_message_prob \
                = self.build_decoder(decoder_input, self.tf_is_training, use_teacher_forcing=self.use_teacher_forcing)
        if self.decoder and self.encoder and self.learning_rate is not None:
            self.train_op, self.tf_output_loss, self.tf_kl_loss \
                = self.build_trainer(self.tf_message_log_prob, self.tf_message,
                                     self.tf_latent_mean, self.tf_latent_log_std, self.learning_rate, variational=variational)

            with tf.name_scope("SAVER"):
                self.saver = tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=10)
        init = tf.global_variables_initializer()
        self.sess = tf.InteractiveSession()
        self.sess.run(init)
        if load_from_save:
            print('Loading from save...')
            load_scope_from_save(save_dir, self.sess, 'LEARNED_EMBEDDINGS')
            if self.encoder:
                load_scope_from_save(save_dir, self.sess, 'MESSAGE_ENCODER')
            if self.decoder:
                load_scope_from_save(save_dir, self.sess, 'MESSAGE_DECODER')

    def encode(self, np_message, batch_size=None):
        """Converts sentences encoded as numpy arrays to points in a latent space."""
        assert self.encoder
        if batch_size is None:
            batch_size = np_message.shape[0]
        all_val_latent_batches = []
        val_batch_gen = chat_model_func.BatchGenerator(np_message, batch_size)
        for batch_index, np_message_batch in enumerate(val_batch_gen.generate_batches()):
            np_val_batch_latent = self.sess.run(self.tf_latent_message, feed_dict={self.tf_message: np_message_batch})
            all_val_latent_batches.append(np_val_batch_latent)
        np_latent_message = np.concatenate(all_val_latent_batches, axis=0)
        return np_latent_message

    def decode(self, np_latent, batch_size=None):
        """Converts points in a latent space to sentences encoded as numpy arrays."""
        assert self.decoder and not self.encoder
        if batch_size is None:
            batch_size = np_latent.shape[0]
        all_val_message_batches = []
        val_batch_gen = chat_model_func.BatchGenerator(np_latent, batch_size)
        for batch_index, np_latent_batch in enumerate(val_batch_gen.generate_batches()):
            np_val_batch_reconstruct = self.sess.run(self.tf_message_prediction, feed_dict={self.tf_latent: np_latent_batch})
            all_val_message_batches.append(np_val_batch_reconstruct)
        np_val_message_reconstruct = np.concatenate(all_val_message_batches, axis=0)
        return np_val_message_reconstruct

    def reconstruct(self, np_input, batch_size=None):
        """Converts sentences into a latent space format, then reconstructs them.
        Returns reconstructions of the input sentences, formatted as numpy arrays."""
        assert self.encoder and self.decoder
        if batch_size is None:
            batch_size = np_input.shape[0]
        all_val_message_batches = []
        val_batch_gen = chat_model_func.BatchGenerator(np_input, batch_size)
        for batch_index, np_message_batch in enumerate(val_batch_gen.generate_batches()):
            np_val_batch_reconstruct = self.sess.run(self.tf_message_prediction, feed_dict={self.tf_message: np_message_batch})
            all_val_message_batches.append(np_val_batch_reconstruct)
        np_val_message_reconstruct = np.concatenate(all_val_message_batches, axis=0)
        return np_val_message_reconstruct

    def train(self, np_input, num_epochs, batch_size, keep_prob=1.0, kl_const_start=0, kl_increase=0.1, kl_max=1.0, verbose=True):
        """Trains on examples from np_input for num_epoch epochs,
        by dividing the data into batches of size batch_size."""
        examples_per_print = 200

        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            print('Epoch: %s' % epoch)
            kl_const = min(kl_const_start + epoch * kl_increase, kl_max)
            if verbose:
                print('KL multiplier: %s' % kl_const)
            train_batch_gen = chat_model_func.BatchGenerator(np_input, batch_size)
            all_train_message_batches = []
            per_print_batch_losses = []
            per_print_batch_kl_losses = []
            for batch_index, np_message_batch in enumerate(train_batch_gen.generate_batches()):
                _, batch_output_loss, batch_kl_loss, np_batch_message_reconstruct = self.sess.run([self.train_op, self.tf_output_loss,
                                                                                                   self.tf_kl_loss, self.tf_message_prediction],
                                                                                                   feed_dict={self.tf_message: np_message_batch,
                                                                                                              self.tf_keep_prob: keep_prob,
                                                                                                              self.tf_kl_const: kl_const,
                                                                                                              self.tf_is_training: True})
                all_train_message_batches.append(np_batch_message_reconstruct)
                per_print_batch_losses.append(batch_output_loss)
                per_print_batch_kl_losses.append(batch_kl_loss)
                if batch_index % examples_per_print == 0 and verbose:
                    print('Batch prediction loss: %s' % np.mean(per_print_batch_losses))
                    print('Batch kl loss: %s' % np.mean(per_print_batch_kl_losses))
                    per_print_batch_losses = []
                    per_print_batch_kl_losses = []
            self.saver.save(self.sess, self.save_dir, global_step=epoch)
            epoch_end_time = time.time()
            print('Epoch elapsed time: %s' % (epoch_end_time - epoch_start_time))

        np_train_message_reconstruct = np.concatenate(all_train_message_batches, axis=0)
        return np_train_message_reconstruct

    def build_encoder(self, tf_message_embs, tf_keep_prob, include_epsilon=True):
        """Build encoder portion of autoencoder in Tensorflow."""
        with tf.variable_scope('MESSAGE_ENCODER'):

            if REVERSE_INPUT_MESSAGES:
                tf_message_embs = tf.reverse(tf_message_embs, axis=[1], name='reverse_message_embs')

            tf_message_embs_dropout = tf.nn.dropout(tf_message_embs, tf_keep_prob)

            message_lstm = tf.contrib.rnn.LSTMCell(num_units=self.rnn_size)
            tf_message_outputs, tf_message_state = tf.nn.dynamic_rnn(message_lstm, tf_message_embs_dropout, dtype=tf.float32)
            tf_last_output = tf_message_outputs[:, -1, :]
            tf_last_output_dropout = tf.nn.dropout(tf_last_output, tf_keep_prob)
            with tf.name_scope('latent_mean'):
                tf_latent_mean, _, _ = baseline_model_func.create_dense_layer(tf_last_output_dropout, self.rnn_size, self.rnn_size)
            with tf.name_scope('latent_std'):
                tf_latent_log_std, _, _ = baseline_model_func.create_dense_layer(tf_last_output_dropout, self.rnn_size, self.rnn_size, activation='relu')
            if include_epsilon:
                tf_epsilon = tf.random_normal(tf.shape(tf_latent_mean), stddev=1, mean=0)
                tf_sampled_latent = tf_latent_mean + tf.exp(tf_latent_log_std) * tf_epsilon
            else:
                tf_sampled_latent = tf_latent_mean
        return tf_sampled_latent, tf_latent_mean, tf_latent_log_std

    def build_decoder(self, tf_latent_input, tf_is_training, use_teacher_forcing=True):
        """Build decoder portion of autoencoder in Tensorflow."""
        # USE_TEACHER_FORCING boolean is not yet implemented!!!
        tf_latent_input_shape = tf.shape(tf_latent_input)
        m = tf_latent_input_shape[0]
        with tf.variable_scope('MESSAGE_DECODER'):
            # tf_decoder_input_tile = tf.tile(tf.reshape(tf_decoder_input, [-1, 1, self.rnn_size]),
            #                                 [1, self.max_message_size, 1])
            output_weight = tf.get_variable('output_weight',
                                            shape=[self.rnn_size, self.word_embedding_size],
                                            initializer=tf.contrib.layers.xavier_initializer())
            output_bias = tf.get_variable('output_bias',
                                          shape=[self.word_embedding_size],
                                          initializer=tf.contrib.layers.xavier_initializer())

            tf_go_token = tf.get_variable('go_token', shape=[1, self.word_embedding_size])
            tf_go_token_tile = tf.tile(tf_go_token, [m, 1])

            response_lstm = tf.contrib.rnn.LSTMCell(num_units=self.rnn_size)
            tf_hidden_state = response_lstm.zero_state(m, tf.float32)
            all_word_logits = []
            all_word_probs = []
            all_word_predictions = []
            for i in range(self.max_message_size):
                if i == 0:
                    tf_teacher_signal = tf_go_token_tile  # give model go token
                else:
                    tf_teacher_true_label = self.tf_message_embs[:, i - 1, :]  # @i=1, selects first label word
                    tf_teacher_test_label = tf_word_prediction_embs
                    tf_teacher_signal = tf.cond(tf_is_training,
                                                lambda: tf_teacher_true_label,
                                                lambda: tf_teacher_test_label)
                if use_teacher_forcing:
                    tf_decoder_input = tf.concat([tf_latent_input, tf_teacher_signal], axis=1)
                else:
                    tf_decoder_input = tf_latent_input

                tf_output, tf_hidden_state = response_lstm(tf_decoder_input, tf_hidden_state)
                tf_word_emb = tf.tanh(tf.matmul(tf_output, output_weight) + output_bias)
                tf_word_logits = tf.matmul(tf_word_emb, self.tf_learned_embeddings, transpose_b=True)
                tf_word_prob = tf.nn.softmax(tf_word_logits)
                tf_word_prediction = tf.argmax(tf_word_logits, axis=1)

                tf_word_prediction_embs = tf.nn.embedding_lookup(self.tf_learned_embeddings, tf_word_prediction)

                all_word_logits.append(tf_word_logits)
                all_word_probs.append(tf_word_prob)
                all_word_predictions.append(tf_word_prediction)

            with tf.name_scope('decoder_outputs'):
                tf_message_logits = tf.stack(all_word_logits, axis=1)
                tf_message_prob = tf.stack(all_word_probs, axis=1)
                tf_message_prediction = tf.stack(all_word_predictions, axis=1)

        return tf_message_prediction, tf_message_logits, tf_message_prob

    def build_trainer(self, tf_message_log_prob, tf_message, tf_latent_mean, tf_latent_log_std, learning_rate, variational=True):
        """Calculate loss function and construct optimizer op
        for 'tf_message_log_prob' prediction and 'tf_message' label."""
        with tf.variable_scope('LOSS'):
            tf_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=tf_message_log_prob,
                                                                       labels=tf_message,
                                                                       name='word_losses')
            tf_output_loss = tf.reduce_mean(tf_losses)
            # Add KL loss
            if variational:
                tf_kl_loss = -tf.reduce_mean(.5 * (1 + tf_latent_log_std - tf.square(tf_latent_mean) - tf.exp(tf_latent_log_std)))
                tf_kl_loss *= self.tf_kl_const
            else:
                tf_kl_loss = tf.zeros(())

            with tf.name_scope('total_loss'):
                tf_total_loss = tf_output_loss + tf_kl_loss

        train_op = tf.train.AdamOptimizer(learning_rate).minimize(tf_total_loss)
        return train_op, tf_output_loss, tf_kl_loss


def load_scope_from_save(save_dir, sess, scope):
    """Load the encoder model variables from checkpoint in save_dir.
    Store them in session sess."""
    variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
    assert len(variables) > 0
    baseline_model_func.restore_model_from_save(save_dir,
                                                var_list=variables, sess=sess)
