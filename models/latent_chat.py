"""Supporting functions, classes and constants for latent_chat_model.py"""
import numpy as np
import tensorflow as tf
from cic.utils import squad_tools as sdt

from cic import config
from cic.models import autoencoder as aef, chat_model, match_lstm

LEARNING_RATE = .00001
NUM_EXAMPLES = None
BATCH_SIZE = 20
NUM_EPOCHS = 100
VALIDATE_INPUTS = False
NUM_LAYERS = 20
KEEP_PROB = .8
CALCULATE_TRAIN_ACCURACY = True


class LatentChatModel:
    def __init__(self, vocab_size, learning_rate, save_dir, ae_save_dir=config.AUTO_ENCODER_MODEL_SAVE_DIR, restore_from_save=False):
        self.vocab_size = vocab_size
        self.learning_rate = learning_rate
        self.save_dir = save_dir
        self.restore_from_save = restore_from_save

        with tf.Graph().as_default():
            self.encoder = aef.AutoEncoder(aef.LEARNED_EMB_DIM, vocab_size,
                                           aef.RNN_HIDDEN_DIM,
                                           aef.MAX_MSG_LEN, encoder=True, decoder=False,
                                           save_dir=ae_save_dir,
                                           load_from_save=True,
                                           learning_rate=aef.LEARNING_RATE,
                                           variational=False)

        with tf.Graph().as_default():
            self.decoder = aef.AutoEncoder(aef.LEARNED_EMB_DIM, vocab_size,
                                           aef.RNN_HIDDEN_DIM,
                                           aef.MAX_MSG_LEN, encoder=False, decoder=True,
                                           save_dir=ae_save_dir,
                                           load_from_save=True,
                                           learning_rate=aef.LEARNING_RATE,
                                           variational=False)

        self.tf_latent_message, self.tf_latent_prediction, self.tf_keep_prob = self.build(NUM_LAYERS)
        self.tf_latent_response, self.tf_total_loss, self.train_op = self.build_trainer(self.tf_latent_prediction)

        self.init = tf.global_variables_initializer()
        self.sess = tf.InteractiveSession()
        self.sess.run(self.init)

        if self.restore_from_save:
            print('Restoring from save...')
            aef.load_scope_from_save(self.save_dir, self.sess, 'LATENT_CHAT_MODEL')

        with tf.name_scope("SAVER"):
            self.saver = tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=10)

    def build(self, num_layers):
        with tf.variable_scope('LATENT_CHAT_MODEL'):
            tf_latent_message = tf.placeholder(tf.float32, shape=(None, aef.RNN_HIDDEN_DIM), name='latent_message')

            m = tf.shape(tf_latent_message)[0]

            padding_size = aef.RNN_HIDDEN_DIM

            tf_padding = tf.zeros([m, padding_size])  # integer division

            tf_keep_prob = tf.placeholder_with_default(1.0, shape=(), name='keep_prob')

            tf_input = tf.concat([tf_latent_message, tf_padding], axis=1)

            input_size = aef.RNN_HIDDEN_DIM + padding_size

            for i in range(num_layers):
                with tf.variable_scope('RESIDUAL_' + str(i)):
                    tf_input_dropout = tf.nn.dropout(tf_input, tf_keep_prob)
                    with tf.variable_scope('INPUT_LAYER'):
                        tf_relu, tf_w1, tf_b1 = match_lstm.create_dense_layer(tf_input_dropout, input_size,
                                                                              input_size,
                                                                              use_xavier=True,
                                                                              activation='relu',
                                                                              std=.0001)
                    with tf.variable_scope('OUTPUT_LAYER'):
                        tf_output, tf_w2, tf_b2 = match_lstm.create_dense_layer(tf_relu, input_size,
                                                                                input_size,
                                                                                use_xavier=True,
                                                                                activation=None,
                                                                                std=.0001)
                    tf_input = tf_input + tf_output

            tf_latent_prediction = tf_input[:, :aef.RNN_HIDDEN_DIM]

        return tf_latent_message, tf_latent_prediction, tf_keep_prob

    def build_trainer(self, tf_latent_prediction):
        tf_latent_response = tf.placeholder(tf.float32, shape=(None, aef.RNN_HIDDEN_DIM), name='latent_response')
        tf_total_loss = tf.reduce_mean(tf.square(tf_latent_response - tf_latent_prediction))
        train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(tf_total_loss)

        return tf_latent_response, tf_total_loss, train_op

    def predict(self, np_latent_message, batch_size):
        latent_batch_gen = chat_model.BatchGenerator([np_latent_message], batch_size)

        all_batch_responses = []
        for np_message_batch in latent_batch_gen.generate_batches():
            assert np_message_batch.shape[0] != 0
            np_batch_response = self.sess.run(self.tf_latent_prediction,
                feed_dict={self.tf_latent_message: np_message_batch})
            all_batch_responses.append(np_batch_response)
        #print(all_batch_responses[0].shape)
        np_model_latent_response = np.concatenate(all_batch_responses, axis=0)

        return np_model_latent_response

    def predict_string(self, your_message, nlp, vocab_dict, vocabulary):
        np_your_message = aef.convert_string_to_numpy(your_message, nlp, vocab_dict)
        np_your_latent_message = self.encoder.encode(np_your_message, aef.BATCH_SIZE)
        np_model_latent_response = self.predict(np_your_latent_message, aef.BATCH_SIZE)
        np_model_response = self.decoder.decode(np_model_latent_response, aef.BATCH_SIZE)
        model_response = sdt.convert_numpy_array_to_strings(np_model_response, vocabulary,
                                                            stop_token=aef.STOP_TOKEN,
                                                            keep_stop_token=True)[0]
        return model_response

    def train(self, np_latent_message, np_latent_response, num_epochs, batch_size, keep_prob):
        for epoch in range(num_epochs):
            print('Epoch: %s' % epoch)
            per_print_losses = []
            latent_batch_gen = chat_model.BatchGenerator([np_latent_message, np_latent_response], batch_size)

            for np_message_batch, np_response_batch in latent_batch_gen.generate_batches():
                assert np_message_batch.shape[0] != 0
                np_batch_loss, np_batch_response, _ = self.sess.run([self.tf_total_loss, self.tf_latent_prediction, self.train_op],
                                                               feed_dict={self.tf_latent_message: np_message_batch,
                                                                          self.tf_latent_response: np_response_batch,
                                                                          self.tf_keep_prob: keep_prob})
                per_print_losses.append(np_batch_loss)

            print('Message std: %s' % np.std(np_message_batch))
            print('Response std: %s' % np.std(np_response_batch))
            print('Prediction std: %s' % np.std(np_batch_response))
            print('Loss: %s' % np.mean(per_print_losses))
            self.saver.save(self.sess, self.save_dir, global_step=epoch)

