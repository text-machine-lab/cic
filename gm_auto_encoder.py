import gmtk
import tensorflow as tf
import cornell_movie_dialogues as cmd
import numpy as np
import squad_dataset_tools as sdt
import baseline_model_func
import pickle
import os


RESTORE_FROM_SAVE = True
SAVE_DIR = './data/autoencoder/first/'


class AutoEncoder(gmtk.GenericModel):
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
                = self.build_decoder(decoder_input, self.inputs['is training'], use_teacher_forcing=True)

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
                tf_latent_mean, _, _ = baseline_model_func.create_dense_layer(tf_last_output_dropout, self.rnn_size,
                                                                              self.rnn_size)
            with tf.name_scope('latent_std'):
                tf_latent_log_std, _, _ = baseline_model_func.create_dense_layer(tf_last_output_dropout,
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

    #     # Load specific scopes from save - if not here, entire Graph is loaded
    #
    #     # Build model with separate decoders for training and prediction
    #     self.inputs['message'] \
    #         = tf.placeholder(dtype=tf.int32, shape=(None, self.max_len), name='message')
    #     self.inputs['code'] \
    #         = tf.placeholder(dtype=tf.float32, shape=(None, self.rnn_size), name='code')
    #     self.outputs['embeddings'] \
    #         = tf.get_variable('embeddings', shape=(self.vocab_size, self.emb_size),
    #                           initializer=tf.contrib.layers.xavier_initializer())
    #     self.inputs['keep prob'] = tf.placeholder_with_default(1.0, shape=(), name='keep_prob')
    #
    #     self.outputs['code'] \
    #         = self.build_encoder(self.inputs['message'], self.outputs['embeddings'], self.inputs['keep prob'])
    #
    #     # For training
    #     with tf.variable_scope('DECODER'):
    #         self.outputs['train_prediction'], tf_train_logits, self.outputs['train_probability'] \
    #             = self.build_decoder(self.outputs['code'], self.outputs['embeddings'], tf_labels=self.inputs['message'])
    #
    #     with tf.variable_scope('LOSS'):
    #         self.loss = self.build_trainer(tf_train_logits, self.inputs['message'])
    #
    #     # For prediction
    #     with tf.variable_scope('DECODER', reuse=True):
    #         self.outputs['prediction'], _, self.outputs['probability'] \
    #             = self.build_decoder(self.inputs['code'], self.outputs['embeddings'], tf_labels=None)
    #
    # def build_encoder(self, tf_message, tf_embeddings, tf_keep_prob, reverse_input_messages=True):
    #     """Build encoder portion of autoencoder."""
    #     tf_message_embs = tf.nn.embedding_lookup(tf_embeddings, tf_message)
    #     if reverse_input_messages:
    #         tf_message_embs = tf.reverse(tf_message_embs, axis=[1], name='reverse_message_embs')
    #     tf_message_embs_dropout = tf.nn.dropout(tf_message_embs, tf_keep_prob)
    #     message_lstm = tf.contrib.rnn.LSTMCell(num_units=self.rnn_size)
    #     tf_message_outputs, tf_message_state = tf.nn.dynamic_rnn(message_lstm, tf_message_embs_dropout, dtype=tf.float32)
    #     tf_last_output = tf_message_outputs[:, -1, :]
    #     tf_last_output_dropout = tf.nn.dropout(tf_last_output, tf_keep_prob)
    #
    #     return tf_last_output_dropout
    #
    # def build_decoder(self, tf_latent_input, tf_embeddings, tf_labels=None):
    #     """Build decoder portion of autoencoder in Tensorflow."""
    #     tf_latent_input_shape = tf.shape(tf_latent_input)
    #     m = tf_latent_input_shape[0]
    #
    #     output_weight = tf.get_variable('output_weight',
    #                                     shape=[self.rnn_size, self.emb_size],
    #                                     initializer=tf.contrib.layers.xavier_initializer())
    #     output_bias = tf.get_variable('output_bias',
    #                                   shape=[self.emb_size],
    #                                   initializer=tf.contrib.layers.xavier_initializer())
    #
    #     tf_go_token = tf.get_variable('go_token', shape=[1, self.emb_size])
    #     tf_go_token_tile = tf.tile(tf_go_token, [m, 1])
    #
    #     tf_label_embs = None
    #     if tf_labels is not None:
    #         tf_label_embs = tf.nn.embedding_lookup(tf_embeddings, tf_labels)
    #
    #     response_lstm = tf.contrib.rnn.LSTMCell(num_units=self.rnn_size)
    #     tf_hidden_state = response_lstm.zero_state(m, tf.float32)
    #     all_word_logits = []
    #     all_word_probs = []
    #     all_word_predictions = []
    #     for i in range(self.max_len):
    #         if i == 0:
    #             tf_teacher_signal = tf_go_token_tile  # give model go token on first step
    #         else:
    #             if tf_labels is not None:
    #                 tf_teacher_signal = tf_label_embs[:, i - 1, :]  # @i=1, selects first label word
    #             else:
    #                 tf_teacher_signal = tf_word_prediction_embs  # available only for i > 0
    #
    #         tf_decoder_input = tf.concat([tf_latent_input, tf_teacher_signal], axis=1)
    #
    #         tf_output, tf_hidden_state = response_lstm(tf_decoder_input, tf_hidden_state)
    #         tf_word_emb = tf.tanh(tf.matmul(tf_output, output_weight) + output_bias)
    #         tf_word_logits = tf.matmul(tf_word_emb, tf_embeddings, transpose_b=True)
    #         tf_word_prob = tf.nn.softmax(tf_word_logits)
    #         tf_word_prediction = tf.argmax(tf_word_logits, axis=1)
    #
    #         tf_word_prediction_embs = tf.nn.embedding_lookup(tf_embeddings, tf_word_prediction)
    #
    #         all_word_logits.append(tf_word_logits)
    #         all_word_probs.append(tf_word_prob)
    #         all_word_predictions.append(tf_word_prediction)
    #
    #     with tf.name_scope('decoder_outputs'):
    #         tf_message_logits = tf.stack(all_word_logits, axis=1)
    #         tf_message_prob = tf.stack(all_word_probs, axis=1)
    #         tf_message_prediction = tf.stack(all_word_predictions, axis=1)
    #
    #     return tf_message_prediction, tf_message_logits, tf_message_prob


    # def build_trainer(self, tf_message_logits, tf_message):
    #     """Calculate loss function and construct optimizer op
    #     for 'tf_message_log_prob' prediction and 'tf_message' label."""
    #     tf_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=tf_message_logits,
    #                                                                labels=tf_message,
    #                                                                name='word_losses')
    #     tf_output_loss = tf.reduce_mean(tf_losses)
    #
    #     return tf_output_loss

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

    def action_per_epoch(self, output_tensor_dict, epoch_index, is_training, **kwargs):
        """Print the loss for debugging purposes."""
        if is_training:
            epoch_loss = np.mean(output_tensor_dict['loss'])
            print('Epoch loss: %s' % epoch_loss)

        return True

# EXECUTION ############################################################################################################

if __name__ == '__main__':
    saved_token_to_id = None
    if RESTORE_FROM_SAVE:
        # Reuse vocabulary when restoring from save
        saved_token_to_id = pickle.load(open(os.path.join(SAVE_DIR, 'vocabulary.pkl'), 'rb'))

    cmd_dataset = cmd.CornellMovieDialoguesDataset(max_message_length=20, token_to_id=saved_token_to_id)

    train_cmd, val_cmd = cmd_dataset.split(fraction=0.9, seed='hello world')

    print('Number of training examples: %s' % len(train_cmd))
    print('Number of validation examples: %s' % len(val_cmd))

    token_to_id, id_to_token = cmd_dataset.get_vocabulary()

    if RESTORE_FROM_SAVE:
        assert saved_token_to_id == token_to_id

    for i in range(10):
        print(id_to_token[i])

    # Save vocabulary
    pickle.dump(token_to_id, open(os.path.join(SAVE_DIR, 'vocabulary.pkl'), 'wb'))

    autoencoder = AutoEncoder(len(token_to_id), tensorboard_name='gmae', save_dir=SAVE_DIR,
                              restore_from_save=RESTORE_FROM_SAVE)

    autoencoder.train(train_cmd, output_tensor_names=['train_prediction'],
                      parameter_dict={'keep prob': 0.9, 'learning rate': .0005},
                      num_epochs=100, batch_size=20, verbose=True)

    def calculate_train_accuracy():
        predictions = autoencoder.decode(autoencoder.encode(train_cmd))

        # Here, I need to convert predictions back to English and print
        reconstructed_messages = sdt.convert_numpy_array_to_strings(predictions, id_to_token,
                                                                    stop_token=cmd_dataset.stop_token,
                                                                    keep_stop_token=True)

        for i in range(10):
            print(' '.join(cmd_dataset.messages[train_cmd.indices[i]]) + " | " + reconstructed_messages[i])

        num_train_correct = 0
        for i in range(len(reconstructed_messages)):
            original_message = ' '.join(cmd_dataset.messages[train_cmd.indices[i]])
            if original_message == reconstructed_messages[i]:
                num_train_correct += 1

        print('Train EM accuracy: %s' % (num_train_correct / len(reconstructed_messages)))


    def predict_using_autoencoder_and_calculate_accuracy():

        val_predictions = autoencoder.decode(autoencoder.encode(val_cmd))

        val_reconstructed_messages = sdt.convert_numpy_array_to_strings(val_predictions, id_to_token,
                                                                        stop_token=cmd_dataset.stop_token,
                                                                        keep_stop_token=True)
        for i in range(10):
            print(' '.join(cmd_dataset.messages[val_cmd.indices[i]]) + " | " + val_reconstructed_messages[i])

        num_val_correct = 0
        for i in range(len(val_reconstructed_messages)):
            original_message = ' '.join(cmd_dataset.messages[val_cmd.indices[i]])
            if original_message == val_reconstructed_messages[i]:
                num_val_correct += 1

        print('Validation EM accuracy: %s' % (num_val_correct / len(val_reconstructed_messages)))


    def input_arbitrary_messages_into_autoencoder():
        print('Testing the autoencoder...')
        # Test autoencoder using stdin
        while True:
            message = input('Message: ')
            np_message = cmd_dataset.convert_strings_to_numpy([message])
            print(np_message)
            np_code = autoencoder.encode(np_message)
            print(np_code[:10])
            np_message_reconstruct = autoencoder.decode(np_code)
            message_reconstruct = cmd_dataset.convert_numpy_to_strings(np_message_reconstruct)[0]
            print('Reconstruct: %s' % message_reconstruct)

    # calculate_train_accuracy()
    print()
    # predict_using_autoencoder_and_calculate_accuracy()
    input_arbitrary_messages_into_autoencoder()





