import numpy as np
import tensorflow as tf

import arcadian.gm
from cic.qa import match_lstm


class AutoEncoder(arcadian.gm.GenericModel):
    def __init__(self, vocab_size, max_len=20, rnn_size=500, emb_size=200, encoder=True, decoder=True, **kwargs):
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.rnn_size = rnn_size
        self.emb_size = emb_size
        self.encoder = encoder
        self.decoder = decoder

        # Define parameters for build before calling main constructor
        super().__init__(**kwargs)

    def build(self):
        self.inputs['message'] = tf.placeholder_with_default(tf.zeros([1, self.max_len], dtype=tf.int32),
                                                      [None, self.max_len], name='input_message')
        self.inputs['code'] = tf.placeholder(dtype=tf.float32, shape=[None, self.rnn_size], name='latent_embedding')
        self.inputs['keep prob'] = tf.placeholder_with_default(1.0, (), name='keep_prob')
        self.tf_kl_const = tf.placeholder_with_default(1.0, (), name='kl_const')

        with tf.variable_scope('LEARNED_EMBEDDINGS'):
            self.tf_learned_embeddings = tf.get_variable('learned_embeddings',
                                                         shape=[self.vocab_size, self.emb_size],
                                                         initializer=tf.contrib.layers.xavier_initializer())
            self.tf_message_embs = tf.nn.embedding_lookup(self.tf_learned_embeddings, self.inputs['message'],
                                                          name='message_embeddings')
        if self.encoder:
            self.outputs['code'], self.tf_latent_mean, self.tf_latent_log_std \
                = self.build_encoder(self.tf_message_embs, self.inputs['keep prob'],
                                     include_epsilon=False)
        if self.decoder:
            if self.encoder:
                decoder_input = self.outputs['code']
            else:
                decoder_input = self.inputs['code']
            self.outputs['prediction'], self.tf_message_log_prob, self.outputs['train_probability'] \
                = self.build_decoder(decoder_input, self.inputs['is_training'], use_teacher_forcing=True)

            self.outputs['train_prediction'] = self.outputs['prediction']

        if self.decoder and self.encoder:
            self.tf_output_loss, self.tf_kl_loss \
                = self.build_trainer(self.tf_message_log_prob, self.inputs['message'],
                                     self.tf_latent_mean, self.tf_latent_log_std)

    def build_encoder(self, tf_message_embs, tf_keep_prob, include_epsilon=True):
        """Build encoder portion of autoencoder in Tensorflow."""
        with tf.variable_scope('MESSAGE_ENCODER'):

            tf_message_embs = tf.reverse(tf_message_embs, axis=[1], name='reverse_message_embs')

            tf_message_embs_dropout = tf.nn.dropout(tf_message_embs, tf_keep_prob)

            message_lstm = tf.contrib.rnn.LSTMCell(num_units=self.rnn_size)
            tf_message_outputs, tf_message_state = tf.nn.dynamic_rnn(message_lstm, tf_message_embs_dropout,
                                                                     dtype=tf.float32)
            tf_last_output = tf_message_outputs[:, -1, :]
            tf_last_output_dropout = tf.nn.dropout(tf_last_output, tf_keep_prob)
            with tf.name_scope('latent_mean'):
                tf_latent_mean, _, _ = match_lstm.create_dense_layer(tf_last_output_dropout, self.rnn_size,
                                                                     self.rnn_size)
            with tf.name_scope('latent_std'):
                tf_latent_log_std, _, _ = match_lstm.create_dense_layer(tf_last_output_dropout,
                                                                        self.rnn_size, self.rnn_size,
                                                                        activation='relu')
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
                                            shape=[self.rnn_size, self.emb_size],
                                            initializer=tf.contrib.layers.xavier_initializer())
            output_bias = tf.get_variable('output_bias',
                                          shape=[self.emb_size],
                                          initializer=tf.contrib.layers.xavier_initializer())

            tf_go_token = tf.get_variable('go_token', shape=[1, self.emb_size])
            tf_go_token_tile = tf.tile(tf_go_token, [m, 1])

            response_lstm = tf.contrib.rnn.LSTMCell(num_units=self.rnn_size)
            tf_hidden_state = response_lstm.zero_state(m, tf.float32)
            all_word_logits = []
            all_word_probs = []
            all_word_predictions = []
            for i in range(self.max_len):
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

    def build_trainer(self, tf_message_log_prob, tf_message, tf_latent_mean, tf_latent_log_std):
        """Calculate loss function and construct optimizer op
        for 'tf_message_log_prob' prediction and 'tf_message' label."""
        with tf.variable_scope('LOSS'):
            tf_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=tf_message_log_prob,
                                                                       labels=tf_message,
                                                                       name='word_losses')
            tf_output_loss = tf.reduce_mean(tf_losses)
            # Add KL loss
            # if variational:
            #     tf_kl_loss = -tf.reduce_mean(
            #         .5 * (1 + tf_latent_log_std - tf.square(tf_latent_mean) - tf.exp(tf_latent_log_std)))
            #     tf_kl_loss *= self.tf_kl_const
            # else:
            tf_kl_loss = tf.zeros(())

            with tf.name_scope('total_loss'):
                tf_total_loss = tf_output_loss + tf_kl_loss

        self.loss = tf_total_loss
        return tf_output_loss, tf_kl_loss

    def encode(self, dataset):
        # Convert to dictionary if numpy input
        if isinstance(dataset, np.ndarray):
            dataset = {'message': dataset}

        results = self.predict(dataset, output_tensor_names=['code'])
        return results['code']

    def decode(self, dataset):
        # Convert to dictionary if numpy input
        if isinstance(dataset, np.ndarray):
            dataset = {'code': dataset}

        results = self.predict(dataset, output_tensor_names=['prediction'])

        return results['prediction']

    def action_per_epoch(self, output_tensor_dict, epoch_index, is_training, parameter_dict, **kwargs):
        """Print the loss for debugging purposes."""
        if is_training:
            epoch_loss = np.mean(output_tensor_dict['loss'])
            print('Epoch loss: %s' % epoch_loss)

        return True

    # def action_before_training(self, placeholder_dict, num_epochs, is_training, output_tensor_names,
    #                            parameter_dict, batch_size=32, **kwargs):
    #     """Optional: Define action to take place at the beginning of training/prediction, once. This could be
    #     used to set output_tensor_names so that certain ops always execute, as needed for other action functions."""
    #     if is_training:
    #         output_tensor_names.append('loss')

    def action_per_batch(self, input_batch_dict, output_batch_dict, epoch_index, batch_index, is_training,
                         parameter_dict, **kwargs):
        """Optional: Define action to take place at the end of every batch. Can use this
        for printing accuracy, saving statistics, etc. Remember, if is_training=False, we are using the model for
        prediction. Check for this."""

        # Save every 1000 batches!
        if batch_index % 1000 == 0 and is_training:
            print('Saving...')
            if self.save_per_epoch and self.trainable and is_training:
                self.saver.save(self.sess, self.save_dir, global_step=epoch_index)

            print('Batch loss: %s' % np.mean(output_batch_dict['loss']))

            if 'validation' in kwargs:
                val_losses = self.predict(kwargs['validation'], output_tensor_names=['loss'])['loss']
                print('Validation loss: %s' % np.mean(np.mean(val_losses)))






