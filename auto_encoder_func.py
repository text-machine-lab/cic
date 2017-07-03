"""Helper classes and functions for auto_encoder_model.py"""
import tensorflow as tf
import baseline_model_func

class AutoEncoder:
    def __init__(self, word_embedding_size, vocab_size, rnn_size, max_message_size, encoder=True, decoder=True, trainable=True):
        self.word_embedding_size = word_embedding_size
        self.encoder = encoder
        self.decoder = decoder
        self.rnn_size = rnn_size
        self.max_message_size = max_message_size
        self.vocab_size = vocab_size
        self.trainable = trainable

        assert encoder or decoder

        # if self.encoder:
        #     self.build_encoder()
        # elif self.decoder:
        #     self.build_decoder()

    def build_encoder(self, tf_message):
        with tf.variable_scope('MESSAGE_ENCODER'):
            tf_learned_embeddings = tf.get_variable('learned_embeddings',
                                                    shape=[self.vocab_size, self.word_embedding_size],
                                                    initializer=tf.contrib.layers.xavier_initializer())

            tf_message_embs = tf.nn.embedding_lookup(tf_learned_embeddings, tf_message, name='message_embeddings')

            message_lstm = tf.contrib.rnn.LSTMCell(num_units=self.rnn_size)
            tf_message_outputs, tf_message_state = tf.nn.dynamic_rnn(message_lstm, tf_message_embs, dtype=tf.float32)
            tf_last_output = tf_message_outputs[:, -1, :]
        return tf_last_output

    def build_decoder(self, tf_decoder_input):
        with tf.variable_scope('MESSAGE_DECODER'):
            tf_message_final_output_tile = tf.tile(tf.reshape(tf_decoder_input, [-1, 1, self.rnn_size]),
                                                   [1, self.max_message_size, 1])

            response_lstm = tf.contrib.rnn.LSTMCell(num_units=self.rnn_size)
            tf_response_outputs, tf_response_state = tf.nn.dynamic_rnn(response_lstm, tf_message_final_output_tile,
                                                                       dtype=tf.float32)
            print('Creating output layer...')
            output_weight = tf.get_variable('output_weight',
                                            shape=[self.rnn_size, self.vocab_size],
                                            initializer=tf.contrib.layers.xavier_initializer())
            output_bias = tf.get_variable('output_bias',
                                          shape=[self.vocab_size],
                                          initializer=tf.contrib.layers.xavier_initializer())
            with tf.name_scope('tf_message_log_probabilities'):
                tf_response_outputs_reshape = tf.reshape(tf_response_outputs, [-1, self.rnn_size])
                tf_message_log_prob = tf.reshape(
                    tf.matmul(tf_response_outputs_reshape, output_weight) + output_bias,
                    [-1, self.max_message_size, self.vocab_size])

            tf_message_prob = tf.nn.softmax(tf_message_log_prob, name='message_probabilities')

            tf_message_prediction = tf.argmax(tf_message_prob, axis=2)

        return tf_message_prediction, tf_message_log_prob, tf_message_prob

    def build_trainer(self, tf_message_log_prob, tf_message):
        with tf.variable_scope('LOSS'):
            tf_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=tf_message_log_prob,
                                                                       labels=tf_message,
                                                                       name='word_losses')
            with tf.name_scope('total_loss'):
                tf_total_loss = tf.reduce_mean(tf_losses)
        return tf_total_loss

    def load_encoder_from_save(self, save_dir, sess):
        encoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='MESSAGE_ENCODER')
        assert len(encoder_vars) > 0
        baseline_model_func.restore_model_from_save(save_dir,
                                                    var_list=encoder_vars, sess=sess)

    def load_decoder_from_save(self, save_dir, sess):
        decoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='MESSAGE_DECODER')
        assert len(decoder_vars) > 0
        baseline_model_func.restore_model_from_save(save_dir,
                                                    var_list=decoder_vars, sess=sess)

    def __call__(self, np_input):
        pass