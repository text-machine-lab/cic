import gmtk
import tensorflow as tf
import cornell_movie_dialogues


class AutoEncoder(gmtk.GenericModel):
    def build(self):
        # Define arguments to model (vocabulary size required!)
        if 'max_length' not in self.params:
            self.params['max_length'] = 30
        if 'rnn_size' not in self.params:
            self.params['rnn_size'] = 400
        if 'emb_size' not in self.params:
            self.params['emb_size'] = 200
        assert 'vocab_size' in self.params

        self.load_scopes = ['ENCODER', 'DECODER']

        # Build model with separate decoders for training and prediction
        self.inputs['message'] \
            = tf.placeholder(dtype=tf.int32, shape=(None, self.params['max_length']), name='message')
        self.inputs['code'] \
            = tf.placeholder(dtype=tf.float32, shape=(None, self.params['rnn_size']), name='code')
        self.outputs['embeddings'] \
            = tf.get_variable('embeddings', shape=(self.params['vocab_size'], self.params['emb_size']),
                              initializer=tf.contrib.layers.xavier_initializer())
        self.inputs['keep prob'] = tf.placeholder_with_default(1.0, shape=(), name='keep_prob')

        self.outputs['code'] \
            = self.build_encoder(self.inputs['message'], self.outputs['embeddings'], self.inputs['keep prob'])

        # For training
        with tf.variable_scope('DECODER'):
            tf_train_prediction, tf_train_logits, tf_train_prob \
                = self.build_decoder(self.outputs['code'], self.outputs['embeddings'], tf_labels=self.inputs['message'])

        with tf.variable_scope('LOSS'):
            self.loss = self.build_trainer(tf_train_logits, self.inputs['message'])

        # For prediction
        with tf.variable_scope('DECODER', reuse=True):
            self.outputs['prediction'], _, self.outputs['probability'] \
                = self.build_decoder(self.inputs['code'], self.outputs['embeddings'], tf_labels=None)

    def build_encoder(self, tf_message, tf_embeddings, tf_keep_prob, reverse_input_messages=True):
        """Build encoder portion of autoencoder."""
        tf_message_embs = tf.nn.embedding_lookup(tf_embeddings, tf_message)
        if reverse_input_messages:
            tf_message_embs = tf.reverse(tf_message_embs, axis=[1], name='reverse_message_embs')
        tf_message_embs_dropout = tf.nn.dropout(tf_message_embs, tf_keep_prob)
        message_lstm = tf.contrib.rnn.LSTMCell(num_units=self.params['rnn_size'])
        tf_message_outputs, tf_message_state = tf.nn.dynamic_rnn(message_lstm, tf_message_embs_dropout, dtype=tf.float32)
        tf_last_output = tf_message_outputs[:, -1, :]
        tf_last_output_dropout = tf.nn.dropout(tf_last_output, tf_keep_prob)

        return tf_last_output_dropout

    def build_decoder(self, tf_latent_input, tf_embeddings, tf_labels=None):
        """Build decoder portion of autoencoder in Tensorflow."""
        tf_latent_input_shape = tf.shape(tf_latent_input)
        m = tf_latent_input_shape[0]

        output_weight = tf.get_variable('output_weight',
                                        shape=[self.params['rnn_size'], self.params['emb_size']],
                                        initializer=tf.contrib.layers.xavier_initializer())
        output_bias = tf.get_variable('output_bias',
                                      shape=[self.params['emb_size']],
                                      initializer=tf.contrib.layers.xavier_initializer())

        tf_go_token = tf.get_variable('go_token', shape=[1, self.params['emb_size']])
        tf_go_token_tile = tf.tile(tf_go_token, [m, 1])

        tf_label_embs = None
        if tf_labels is not None:
            tf_label_embs = tf.nn.embedding_lookup(tf_embeddings, tf_labels)

        response_lstm = tf.contrib.rnn.LSTMCell(num_units=self.params['rnn_size'])
        tf_hidden_state = response_lstm.zero_state(m, tf.float32)
        all_word_logits = []
        all_word_probs = []
        all_word_predictions = []
        for i in range(self.params['max_length']):

            if i == 0:
                tf_teacher_signal = tf_go_token_tile  # give model go token on first step
            else:
                if tf_labels is not None:
                    tf_teacher_signal = tf_label_embs[:, i - 1, :]  # @i=1, selects first label word
                else:
                    tf_teacher_signal = tf_word_prediction_embs  # available only for i > 0

            tf_decoder_input = tf.concat([tf_latent_input, tf_teacher_signal], axis=1)

            tf_output, tf_hidden_state = response_lstm(tf_decoder_input, tf_hidden_state)
            tf_word_emb = tf.tanh(tf.matmul(tf_output, output_weight) + output_bias)
            tf_word_logits = tf.matmul(tf_word_emb, tf_embeddings, transpose_b=True)
            tf_word_prob = tf.nn.softmax(tf_word_logits)
            tf_word_prediction = tf.argmax(tf_word_logits, axis=1)

            tf_word_prediction_embs = tf.nn.embedding_lookup(tf_embeddings, tf_word_prediction)

            all_word_logits.append(tf_word_logits)
            all_word_probs.append(tf_word_prob)
            all_word_predictions.append(tf_word_prediction)

        with tf.name_scope('decoder_outputs'):
            tf_message_logits = tf.stack(all_word_logits, axis=1)
            tf_message_prob = tf.stack(all_word_probs, axis=1)
            tf_message_prediction = tf.stack(all_word_predictions, axis=1)

        return tf_message_prediction, tf_message_logits, tf_message_prob

    def build_trainer(self, tf_message_logits, tf_message):
        """Calculate loss function and construct optimizer op
        for 'tf_message_log_prob' prediction and 'tf_message' label."""
        tf_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=tf_message_logits,
                                                                   labels=tf_message,
                                                                   name='word_losses')
        tf_output_loss = tf.reduce_mean(tf_losses)

        return tf_output_loss

if __name__ == '__main__':
    cmd_dataset = cornell_movie_dialogues.CornellMovieDialoguesDataset()
    token_to_id, id_to_token = cmd_dataset.get_vocabulary()
    autoencoder = AutoEncoder(vocab_size=len(token_to_id))

    autoencoder.train(cmd_dataset, output_tensor_names=[], num_epochs=1)


