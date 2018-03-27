"""Model which trains on x, y pairs."""
from arcadian.gm import GenericModel
from arcadian.dataset import DictionaryDataset
import tensorflow as tf
from cic.models.rnet_gan import build_linear_layer
import numpy as np

class Seq2Seq(GenericModel):
    def __init__(self, max_s_len, vocab_len, emb_size, rnn_size, **kwargs):

        self.max_s_len = max_s_len  # maximum sequence length for both input and output
        self.vocab_len = vocab_len  # number of vocab words present input, number of embs to learn
        self.emb_size = emb_size  # size of word embeddings
        self.rnn_size = rnn_size  # cell size of LSTM for both encoder and decoder
        self.cell = None

        super().__init__(**kwargs)

    def build(self):

        # initialize placeholders
        inputs = tf.placeholder(tf.int32, shape=(None, self.max_s_len), name='x')

        input_codes = tf.placeholder(tf.float32, shape=(None, self.rnn_size), name='codes')
        word = tf.placeholder(tf.int32, shape=(None,), name='input_word')
        input_state = tf.placeholder(tf.float32, shape=(None, self.rnn_size * 2), name='input_state')
        keep_prob = tf.placeholder_with_default(1.0, shape=(), name='keep_prob')

        with tf.variable_scope('EMBEDDINGS'):
            embs = tf.get_variable('embs', shape=(self.vocab_len, self.emb_size))
            input_embs = tf.nn.embedding_lookup(embs, inputs)

        word_emb = tf.nn.embedding_lookup(embs, word)
        input_word_emb = tf.placeholder(tf.float32, shape=(None, self.emb_size), name='input_word_emb')

        with tf.variable_scope('ENCODER'):
            codes = self.encoder(input_embs)
            dropout_codes = tf.nn.dropout(codes, keep_prob)

        labels = tf.placeholder(tf.int32, shape=(None, self.max_s_len), name='y')
        label_embs = tf.nn.embedding_lookup(embs, labels)

        with tf.variable_scope('DECODER') as s:
            self.cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.rnn_size, state_is_tuple=False)

            preds, probs, logits, zero_state, go_token = self.train_decoder(dropout_codes, label_embs)
            s.reuse_variables()
            pred, prob, logit, state = self.pred_decoder(input_codes, input_word_emb, input_state)

        self.trainer(logits, labels)

        self.load_scopes = ['EMBEDDINGS', 'ENCODER', 'DECODER']

        self.i.update({'message': inputs, 'response': labels, 'word': word, 'keep_prob': keep_prob,
                       'state': input_state, 'code': input_codes, 'input_word_emb': input_word_emb})

        self.o.update({'preds': preds, 'probs': probs, 'logits': logits, 'zero_state': zero_state,
                       'go_token': go_token, 'word_pred': pred, 'word_prob': prob,
                       'word_logit': logit, 'word_state': state, 'word_emb': word_emb,
                       'code': codes})

    def encoder(self, input_embs):

        batch_size = tf.shape(input_embs)[0]

        cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.rnn_size)

        state = cell.zero_state(batch_size, dtype=tf.float32)

        outputs, state = tf.nn.dynamic_rnn(cell, input_embs, initial_state=state,
                                           dtype=tf.float32)

        return outputs[:, -1, :]

    def train_decoder(self, codes, label_embs):
        """Creates a decoder which can be used for training on target labels."""

        # Create LSTM
        batch_size = tf.shape(codes)[0]

        zero_state = self.cell.zero_state(batch_size, dtype=tf.float32)

        # Create learnable go token to represent start of decoder
        go_token = tf.get_variable('go', shape=(1, self.emb_size))
        go_tile = tf.tile(go_token, [batch_size, 1])

        decoder_inputs = []
        for i in range(self.max_s_len):
            if i == 0:
                each_input = tf.concat([go_tile, codes], axis=-1)
            else:
                each_input = tf.concat([label_embs[:, i-1, :], codes], axis=-1)

            decoder_inputs.append(each_input)

        state = zero_state
        outputs = []
        for input_ in decoder_inputs:
            output, state = self.cell(input_, state)
            outputs.append(output)

        # outputs, state = tf.nn.static_rnn(cell, decoder_inputs, initial_state=zero_state,
        #                                    dtype=tf.float32)

        outputs = tf.stack(outputs, axis=1)

        outputs_flat = tf.reshape(outputs, [-1, self.rnn_size])

        logits_flat = build_linear_layer('output', outputs_flat, self.vocab_len,
                                        xavier=True)

        logits = tf.reshape(logits_flat, [-1, self.max_s_len, self.vocab_len])
        probs = tf.nn.softmax(logits)
        preds = tf.argmax(probs, axis=-1)

        return preds, probs, logits, zero_state, go_token

    def pred_decoder(self, codes, word_emb, state):
        """This creates a decoder which can run word by word."""

        decode_input = tf.concat([word_emb, codes], axis=1)

        output, state = self.cell(decode_input, state)

        logit = build_linear_layer('output', output, self.vocab_len, xavier=True)
        prob = tf.nn.softmax(logit)
        pred = tf.argmax(prob, axis=-1)

        return pred, prob, logit, state

    def trainer(self, outputs, labels):

        ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=outputs)

        self.loss = tf.reduce_mean(ce)

        return self.loss

    def generate_responses(self, msgs, n=5):
        """For a batch of input (messages), generate for each
        input an output (response).

            msgs - a numpy array or Dataset of 'message' features
            n - only sample n highest probability words at each
            timestep

        Returns: a numpy array of responses per input message"""

        # Allow Datasets and numpy arrays as input
        if isinstance(msgs, np.ndarray):
            msgs = DictionaryDataset({'message': msgs})

        codes = self.predict(msgs, outputs=['code'])

        return self.generate_responses_from_codes(codes, n=n)


    def generate_responses_from_codes(self, codes, n=5):

        go_token = self.predict(None, outputs=['go_token'])

        init_hidden = np.zeros([1, self.rnn_size * 2])

        num_samples = codes.shape[0]

        prev_word_embs = np.repeat(go_token, num_samples, axis=0)
        hidden_s = np.repeat(init_hidden, num_samples, axis=0)

        # print('Shape hidden_s: %s' % str(hidden_s.shape))

        sampled_sentence_words = []
        for t in range(self.max_s_len):
            result = self.predict({'state': hidden_s, 'input_word_emb': prev_word_embs, 'code': codes},
                                  outputs=['word_prob', 'word_state'])

            word_probs = result['word_prob']
            hidden_s = result['word_state']

            np_words = np.zeros([num_samples])
            for ex_index in range(word_probs.shape[0]):  # example index
                prob = word_probs[ex_index, :]

                # Look for n words with highest probability
                max_prob_words = []
                for index in range(n):
                    # Find highest prob index
                    max_index = np.argmax(prob)
                    max_prob = prob[max_index]

                    max_prob_words.append((max_index, max_prob))
                    prob[max_index] = 0  # get rid of max prob word

                assert len(max_prob_words) == n

                max_indices = [each_index for (each_index, each_prob) in max_prob_words]
                max_probs = [each_prob for (each_index, each_prob) in max_prob_words]

                # Scale probs into range
                max_probs = max_probs / np.sum(max_probs)
                word_index = np.random.choice(max_indices, p=max_probs)

                np_words[ex_index] = word_index

            # grab embedding per word
            np_word_embs = self.predict({'word': np_words}, outputs=['word_emb'])

            # set as next teacher signal and save word index
            prev_word_embs = np_word_embs
            sampled_sentence_words.append(np_words)

        np_messages = np.stack(sampled_sentence_words, axis=1)

        return np_messages
