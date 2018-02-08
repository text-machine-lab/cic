"""GAN model trained on text represented using embeddings taken from an external source. The intended
source is word embeddings from a fully-trained neural language model."""

import arcadian
import tensorflow as tf
import numpy as np
from cic.models.sgan import get_variables_of_scope, build_linear_layer

class ExtEmbGAN(arcadian.gm.GenericModel):
    def __init__(self, embs, max_s_len, rnn_size, num_dsc_trains, **kwargs):

        self.input_embs = embs  # size of external word embeddings
        self.emb_size = embs.shape[1]
        self.vocab_size = embs.shape[0]
        self.max_s_len = max_s_len  # max sentence length
        self.rnn_size = rnn_size  # cell state size of LSTM generator and discriminator
        self.num_dsc_trains = num_dsc_trains  # number of times to update discriminator, per generator update

        super().__init__(**kwargs)

    def build(self):
        """We build a GAN model which represents words using embeddings taken from an external source.
        First the embeddings must be loaded using an assign op. Then sentences must be given as input for
        training. Random vectors must be given as input to the generator for it to produce random outputs.
        The trainer uses the Improved Wasserstein GAN implementation. The interface allows for loading of
        embeddings, training on real and fake sentences, and generation of fake sentences."""

        # Load embeddings from source
        embs = tf.Variable(self.input_embs, trainable=False, name='embs')

        # # Generator uses internal embeddings to select words
        # with tf.variable_scope('INTERNAL_EMBEDDDINGS'):
        #     internal_embs = tf.get_variable('internal_embs', shape=self.input_embs.shape,
        #                                     initializer=tf.random_normal_initializer(stddev=0.1))

        # Construct input sentences placeholder
        sentences = tf.placeholder(tf.int32, shape=(None, self.max_s_len), name='sentences')
        sentence_embs = tf.nn.embedding_lookup(embs, sentences)

        # Build generator
        z = tf.placeholder(tf.float32, shape=(None, self.rnn_size), name='z')

        with tf.variable_scope('GENERATOR'):

            gen_embs, gen_logits, gen_sentences = self.generator(z, embs)

            # norm_embs = tf.nn.l2_normalize(embs, -1)
            #
            # flat_sentence_embs = tf.reshape(sentence_embs, [-1, self.emb_size])
            # flat_sentence_logits = tf.matmul(flat_sentence_embs, norm_embs, transpose_b=True)
            # sentence_logits = tf.reshape(flat_sentence_logits, [-1, self.max_s_len, self.vocab_size])
            #
            # sentence_reconst = tf.argmax(sentence_logits, axis=-1)
            # gen_sentences = sentence_reconst

        # Build discriminator
        with tf.variable_scope('DISCRIMINATOR') as s:
            real_scores = self.discriminator(sentence_embs)

            s.reuse_variables()

            fake_scores = self.discriminator(gen_embs)

        # Build trainer
        gen_loss, dsc_loss, dsc_real_loss, dsc_fake_loss, gen_lr, dsc_lr = \
            self.trainer(real_scores, fake_scores, sentence_embs, gen_embs)

        # Define variable scopes to save and load
        self.load_scopes = ['GENERATOR', 'DISCRIMINATOR']

        # Define interface to model
        self.i.update({'message': sentences, 'z': z, 'gen_lr': gen_lr, 'dsc_lr': dsc_lr})

        self.o.update({'gen_loss': gen_loss, 'dsc_loss': dsc_loss,
                       'dsc_real_loss': dsc_real_loss, 'dsc_fake_loss': dsc_fake_loss,
                       'real_scores': real_scores, 'fake_scores': fake_scores,
                       'gen_embs': gen_embs, 'gen_logits': gen_logits, 'gen_sentences': gen_sentences,
                       'embs': embs, 'sentence_embs': sentence_embs})

    def trainer(self, real_scores, fake_scores, real_embs, fake_embs):
        """Add optimizers to optimizer list for training both generator and
        discriminator."""
        gen_vars = get_variables_of_scope('GENERATOR')
        dsc_vars = get_variables_of_scope('DISCRIMINATOR')

        assert len(gen_vars) > 0
        assert len(dsc_vars) > 0

        # Norm clipping as suggested in 'Improved Training of Wasserstein GANs'
        # batch_size = tf.shape(real_embs)[0]
        # LAMBDA = 10
        #
        # # We want to regularize the discriminator to become a K lipschitz function!
        # # The paper by default uses 1-lipschitz
        # K = 0.1
        #
        # alpha = tf.random_uniform(
        #     shape=[batch_size, self.max_s_len, 1],
        #     minval=0.,
        #     maxval=1.
        # )
        # differences = fake_embs - real_embs
        # interpolates = real_embs + (alpha * differences)
        #
        # with tf.variable_scope('DISCRIMINATOR') as s:
        #     s.reuse_variables()
        #
        #     intpol_logits = self.discriminator(interpolates)
        #
        # gradients = tf.gradients(intpol_logits, [interpolates])[0]
        # slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
        # gradient_penalty = tf.reduce_mean((slopes - K) ** 2)
        # norm_loss = LAMBDA * gradient_penalty
        #
        # grad_norm = tf.reduce_mean(slopes)

        # Wasserstein losses
        dsc_real_loss = -tf.reduce_mean(real_scores)
        dsc_fake_loss = tf.reduce_mean(fake_scores)

        dsc_loss = dsc_real_loss + dsc_fake_loss  # + norm_loss
        gen_loss = -dsc_fake_loss

        # Add weight clipping from Wasserstein GAN implementation
        clip_op = tf.group(*[p.assign(tf.clip_by_value(p, -0.1, 0.1)) for p in dsc_vars])

        gen_lr = tf.placeholder_with_default(.001, shape=(), name='gen_lr')
        dsc_lr = tf.placeholder_with_default(.001, shape=(), name='dsc_lr')

        # Create optimizers for generator and discriminator.
        gen_op = tf.train.AdamOptimizer(gen_lr).minimize(gen_loss, var_list=gen_vars)
        dsc_op = tf.train.AdamOptimizer(dsc_lr).minimize(dsc_loss, var_list=dsc_vars)

        self.train_ops = [dsc_op] * self.num_dsc_trains + [clip_op] + [gen_op]

        return gen_loss, dsc_loss, dsc_real_loss, dsc_fake_loss, gen_lr, dsc_lr

    def generator(self, z, embs):
        """Uses a random input vector z to generate output word embeddings for a sentence.

        Returns (m x t x e) embeddings for sentences, (m x t x v) scores over vocab for sentence, and
        (m x t) indices of words which make up a sentence (for m batch size, t max sentence length, e
        word embedding size, and v vocabulary size)."""

        batch_size = tf.shape(z)[0]

        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.rnn_size)

        initial_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)

        # Insert random vector z at all timesteps of generator LSTM
        rand_input = tf.tile(tf.reshape(z, [-1, 1, self.rnn_size]), [1, self.max_s_len, 1])

        outputs, state = tf.nn.dynamic_rnn(lstm_cell, rand_input,
                                           initial_state=initial_state,
                                           dtype=tf.float32)

        flat_outputs = tf.reshape(outputs, [-1, self.rnn_size])

        # We produce a probability distribution over the vocabulary from the output of the RNN
        # We the produce embeddings as a weighted sum over the vocabulary.
        # This forces the generator to use the external embeddings as an interface to the discriminator
        flat_logits = build_linear_layer('gen_probs', flat_outputs, self.vocab_size, xavier=True)
        flat_probs = tf.nn.softmax(flat_logits)
        flat_gen_embs = tf.matmul(flat_probs, embs)

        # flat_gen_embs = build_linear_layer('gen_output', flat_outputs, self.emb_size, xavier=True)
        # flat_gen_logits = tf.matmul(flat_gen_embs, embs, transpose_b=True)
        #

        gen_embs = tf.reshape(flat_gen_embs, [-1, self.max_s_len, self.emb_size])
        gen_logits = tf.reshape(flat_logits, [-1, self.max_s_len, self.vocab_size])
        gen_sentences = tf.argmax(gen_logits, axis=-1)

        return gen_embs, gen_logits, gen_sentences

    def discriminator(self, s_embs):
        """Outputs scores for input sentences to indicate how real they are."""

        batch_size = tf.shape(s_embs)[0]

        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.rnn_size)

        initial_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)

        lstm_inputs = [s_embs[:, i, :] for i in range(self.max_s_len)]

        outputs, state = tf.nn.static_rnn(lstm_cell, lstm_inputs,
                                           initial_state=initial_state,
                                           dtype=tf.float32)

        outputs = tf.stack(outputs, axis=1)

        scores = build_linear_layer('gen_output', outputs[:, -1, :], 1, xavier=True)

        return scores

    def action_before_training(self, placeholder_dict, num_epochs, is_training, output_tensor_names,
                               params, batch_size=32, **kwargs):
        """Optional: Define action to take place at the beginning of training/prediction, once. This could be
        used to set output_tensor_names so that certain ops always execute, as needed for other action functions."""
        if is_training:
            output_tensor_names.extend(['gen_loss', 'dsc_loss', 'real_scores', 'fake_scores',
                                        'gen_embs', 'sentence_embs', 'embs'])

    def action_per_batch(self, input_batch_dict, output_batch_dict, epoch_index, batch_index, is_training,
                         params, **kwargs):
        """Optional: Define action to take place at the end of every batch. Can use this
        for printing accuracy, saving statistics, etc. Remember, if is_training=False, we are using the model for
        prediction. Check for this."""
        if ((batch_index % 100 == 0 and batch_index <= 1000) or (batch_index % 1000 == 0)) and is_training:
            print()
            print('Generator Loss: %s' % output_batch_dict['gen_loss'])
            print('Discriminator Loss: %s' % output_batch_dict['dsc_loss'])

            # print('Grad norm: %s' % output_batch_dict['grad_norm'])
            # print('norm_loss: %s' % output_batch_dict['norm_loss'])

            print('Real score mean: %s' % np.mean(output_batch_dict['real_scores']))
            print('Real data mean: %s' % np.mean(output_batch_dict['sentence_embs']))
            print('Real data var: %s' % np.var(output_batch_dict['sentence_embs']))

            print('Fake score mean: %s' % np.mean(output_batch_dict['fake_scores']))
            print('Fake data mean: %s' % np.mean(output_batch_dict['gen_embs']))
            print('Fake data var: %s' % np.var(output_batch_dict['gen_embs']))

            print('Mean embs: %s' % np.mean(output_batch_dict['embs']))
            print('Var embs: %s' % np.var(output_batch_dict['embs']))

        # Save every 1000 batches!
        if batch_index != 0 and batch_index % 100 == 0 and is_training:
            print('Saving...')
            if self.save_per_epoch and self.trainable and is_training:
                self.saver.save(self.sess, self.save_dir, global_step=epoch_index, write_meta_graph=False)