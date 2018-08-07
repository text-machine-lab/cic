"""Model which trains on x, y pairs."""
from arcadian.gm import GenericModel
from arcadian.dataset import DictionaryDataset
import tensorflow as tf
from cic.models.rnet_gan import build_linear_layer
import numpy as np

class Seq2Seq(GenericModel):
    def __init__(self, in_len, out_len, vocab_len, emb_size, rnn_size, attention=False, **kwargs):

        self.in_len = in_len  # max length of input sequence
        self.out_len = out_len  # max length of output sequence
        self.vocab_len = vocab_len  # number of vocab words present input, number of embs to learn
        self.emb_size = emb_size  # size of word embeddings
        self.rnn_size = rnn_size  # cell size of LSTM for both encoder and decoder
        self.cell = None
        self.attention = attention

        super().__init__(**kwargs)

    def build(self):
        """Build a sequence-to-sequence model with optional attention."""

        ################## Create inputs ##########################################

        inputs, input_lens, input_codes, word, \
        input_state, keep_prob, input_word_emb, labels = self.construct_inputs()

        with tf.variable_scope('EMBEDDINGS'):
            embs = tf.get_variable('embs', shape=(self.vocab_len, self.emb_size))
            input_embs = tf.nn.embedding_lookup(embs, inputs)

        word_emb = tf.nn.embedding_lookup(embs, word)
        label_embs = tf.nn.embedding_lookup(embs, labels)

        ################### Create encoder/decoder ##################################

        with tf.variable_scope('ENCODER'):
            enc_states = encoder(input_embs, self.rnn_size)

            # apply dropout
            final_state = enc_states[:, -1, :]  # code is last state
            drop_final_state = tf.nn.dropout(final_state, keep_prob)

        if self.attention:
            dec_input = enc_states
        else:
            dec_input = drop_final_state

        with tf.variable_scope('DECODER') as s:
            self.cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.rnn_size, state_is_tuple=False)

            preds, probs, logits, go_token, zero_state \
                = self.train_decoder(dec_input, label_embs, attention=self.attention)

            s.reuse_variables()

            pred, prob, logit, state = self.pred_decoder(input_codes, input_word_emb, input_state, attention=self.attention)

        self.trainer(logits, labels)

        ######################## Create Interface ######################################

        if self.attention:
            output_code = enc_states
        else:
            output_code = final_state

        self.load_scopes = ['EMBEDDINGS', 'ENCODER', 'DECODER']
        self.i.update({'message': inputs, 'response': labels, 'word': word, 'keep_prob': keep_prob,
                       'state': input_state, 'code': input_codes, 'input_word_emb': input_word_emb})

        self.o.update({'preds': preds, 'probs': probs, 'logits': logits, 'zero_state': zero_state,
                       'go_token': go_token, 'word_pred': pred, 'word_prob': prob,
                       'word_logit': logit, 'word_state': state, 'word_emb': word_emb,
                       'code': output_code})

    def construct_inputs(self):
        """Create placeholders for sequence to sequence model."""
        inputs = tf.placeholder(tf.int32, shape=(None, self.in_len), name='x')
        input_lens = tf.count_nonzero(inputs, axis=1)

        if self.attention:
            input_codes = tf.placeholder(tf.float32, shape=(None, self.in_len, self.rnn_size), name='codes')
        else:
            input_codes = tf.placeholder(tf.float32, shape=(None, self.rnn_size), name='codes')

        word = tf.placeholder(tf.int32, shape=(None,), name='input_word')
        input_state = tf.placeholder(tf.float32, shape=(None, self.rnn_size * 2), name='input_state')
        keep_prob = tf.placeholder_with_default(1.0, shape=(), name='keep_prob')

        input_word_emb = tf.placeholder(tf.float32, shape=(None, self.emb_size), name='input_word_emb')

        labels = tf.placeholder(tf.int32, shape=(None, self.out_len), name='y')

        return inputs, input_lens, input_codes, word, input_state, keep_prob, input_word_emb, labels

    def train_decoder(self, codes, label_embs, attention=False):
        """Creates a decoder which can be used for training on target labels.

        codes - input to decoder as initial cell state
                (with attention this is all encoder states, without attention only last state)
        label_embs - target labels for sequence generation
        attention - whether or not to use attention"""

        # Create LSTM
        batch_size = tf.shape(codes)[0]

        # Create learnable go token to represent start of decoder
        go_token = tf.get_variable('go', shape=(1, self.emb_size))
        go_tile = tf.tile(go_token, [batch_size, 1])

        word_emb_input = []
        for i in range(self.out_len):
            if i == 0:
                each_input = go_tile
            else:
                each_input = label_embs[:, i - 1, :]
            word_emb_input.append(each_input)

        state = self.cell.zero_state(batch_size, tf.float32)

        zero_state = state
        outputs = []

        with tf.variable_scope('RNN') as s:
            for index, word_emb in enumerate(word_emb_input):
                if index > 0:
                    s.reuse_variables()
                # add final enc state or attention over enc states as input
                if attention:
                    context = add_attention(codes, state)
                else:
                    context = codes

                dec_input = tf.concat([word_emb, context], axis=1)

                output, state = self.cell(dec_input, state)
                outputs.append(output)
        outputs = tf.stack(outputs, axis=1)

        outputs_flat = tf.reshape(outputs, [-1, self.rnn_size])
        logits_flat = build_linear_layer('output', outputs_flat, self.vocab_len,
                                         xavier=True)
        logits = tf.reshape(logits_flat, [-1, self.out_len, self.vocab_len])
        probs = tf.nn.softmax(logits)
        preds = tf.argmax(probs, axis=-1)

        return preds, probs, logits, go_token, zero_state

    def pred_decoder(self, codes, word_emb, input_state, attention=False):
        """Runs a single step of the decoder LSTM for prediction.

        codes - input to decoder (all encoder states with attention, last encoder state without)
        word_emb - word_emb of previous prediction as input
        input_state - concatenation of cell and hidden states from previous step
        attention - boolean, whether or not to use attention (changes codes shape)

        Returns: the predicted word index, probabilities over vocabulary, pre-softmax scores over vocabulary,
        final state."""

        with tf.variable_scope('RNN'):
            if attention:
                context = add_attention(codes, input_state)
            else:
                context = codes

            dec_input = tf.concat([word_emb, context], axis=1)

            output, state = self.cell(dec_input, input_state)
        logit = build_linear_layer('output', output, self.vocab_len, xavier=True)
        prob = tf.nn.softmax(logit)
        pred = tf.argmax(prob, axis=-1)

        return pred, prob, logit, state

    def trainer(self, outputs, labels):
        """Cross-entropy loss calculated over outputs using labels.

        Returns: a Tensor containing the computed loss."""
        ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=outputs)
        self.loss = tf.reduce_mean(ce)

        return self.loss

    def generate_responses(self, msgs, n=1):
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
        """Sample responses generated from input codes.

        n - sample from top n highest probability words

        Returns: Numpy array of generated responses. """

        go_token = self.predict(None, outputs=['go_token'])

        init_hidden = np.zeros([1, self.rnn_size * 2])  # this could be a problem!!!!

        num_samples = codes.shape[0]

        prev_word_embs = np.repeat(go_token, num_samples, axis=0)
        hidden_s = np.repeat(init_hidden, num_samples, axis=0)

        # print('Shape hidden_s: %s' % str(hidden_s.shape))

        sampled_sentence_words = []
        for t in range(self.out_len):
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


def encoder(input_embs, rnn_size):
    """Build encoder LSTM on input word embeddings.

    input_embs      - m x t x e Tensor for m utterances of max length t
                      with embedding size e
    rnn_size        - size of rnn cell state

    Returns: hidden states of each time step
    """

    batch_size = tf.shape(input_embs)[0]

    cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=rnn_size)

    state = cell.zero_state(batch_size, dtype=tf.float32)

    outputs, state = tf.nn.dynamic_rnn(cell, input_embs, initial_state=state,
                                       dtype=tf.float32)

    return outputs


def add_attention(enc_states, dec_state):
    """Apply Bahdanau attention to encoder states,
    conditioned on previous hidden state. From
    this paper: https://arxiv.org/pdf/1409.0473.pdf

    enc_states - states from encoder (can be any memory)
                 3-dimensional for axis=1 as axis to apply attention over
    dec_state - previous state of decoder (can be any conditioning on attention)
                2-dimensional

    Returns: Weighted attention over encoder states.
    """

    # determine shapes
    shape = enc_states.get_shape()
    enc_size = shape[-1]
    n_steps = shape[-2]
    dec_size = dec_state.get_shape()[-1]

    # create variables
    xavier = tf.contrib.layers.xavier_initializer()
    u = tf.get_variable('U', shape=(enc_size, enc_size), initializer=xavier)
    w = tf.get_variable('W', shape=(dec_size, enc_size), initializer=xavier)
    v = tf.get_variable('V', shape=(enc_size, 1), initializer=xavier)

    # compute attention score for all encoder states
    lst_enc_states = tf.unstack(enc_states, axis=-2)
    lst_scores = []
    for enc_state in lst_enc_states:
        x = tf.matmul(dec_state, w)
        y = tf.matmul(enc_state, u)
        z = tf.tanh(x + y)
        e = tf.matmul(z, v)
        lst_scores.append(e)

    # compute attention weights
    es = tf.stack(lst_scores, axis=1)
    att = tf.nn.softmax(es, axis=1)

    # computed weighted average of states
    weighted = tf.multiply(enc_states, att)
    c = tf.reduce_sum(weighted, axis=1)
    return c

def create_concat_state(codes_):
    shape = tf.shape(codes_)
    batch_size = shape[0]
    rnn_size = shape[-1]
    zeros = tf.zeros([batch_size, rnn_size])
    concat_state = tf.concat([codes_, zeros], axis=1)
    return concat_state



















