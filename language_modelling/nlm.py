"""Neural language model for generating sentences."""

from arcadian.gm import GenericModel
import tensorflow as tf

class NeuralLanguageModel(GenericModel):
    def __init__(self, max_len, vocab_len, emb_size, hidden_size, **kwargs):
        """Neural language model used to generate English sentences."""

        self.max_len = max_len  # maximum sentence length
        self.vocab_len = vocab_len  # number of vocabulary words
        self.emb_size = emb_size  # size of each word embedding
        self.hidden_size = hidden_size  # size of the LSTM cell

        super().__init__(**kwargs)

    def build(self):
        """Implement Tensorflow model and specify model placeholders and output tensors you wish to evaluate
        using self.input_placeholders and self.output_tensors dictionaries. Specify each entry as a name:tensor pair.
        Specify variable scopes to restore by adding to self.load_scopes list. Specify loss function to train on
        by assigning loss tensor to self.loss variable. Read initialize_loss() documentation for adaptive
        learning rates and evaluating loss tensor at runtime."""

        # Specify inputs
        self.inputs['message'] = tf.placeholder(dtype=tf.int32, shape=(None, self.max_len)) # label
        batch_size = tf.shape(self.inputs['message'])[0]

        # Build embedding matrix
        self.inputs['word_embs'] = tf.get_variable('embs', (self.vocab_len, self.emb_size),
                                                   initializer=tf.random_normal_initializer())


    def _build_rnn_for_training(self, batch_size):
        # Build LSTM decoder
        cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size)
        state = cell.zero_state(batch_size, dtype=tf.float32)

        label_embs = tf.nn.embedding_lookup(self.inputs['word_embs'], self.inputs['message'])

        outputs, state = tf.nn.dynamic_rnn(cell, label_embs, initial_state=state, dtype=tf.float32)

        logits = build_linear_layer('logits', outputs, self.vocab_len)


def build_linear_layer(name, input_tensor, output_size):
    """Build linear layer by creating random weight matrix and bias vector,
    and applying them to input. Weights initialized with random normal
    initializer.

    Arguments:
        - name: Required for unique Variable names
        - input: (num_examples x layer_size) matrix input
        - output_size: size of output Tensor

    Returns: Output Tensor of linear layer with size (num_examples, out_size).
        """

    input_size = input_tensor.get_shape()[-1]  #tf.shape(input_tensor)[1]
    with tf.variable_scope(name):
        scale_w = tf.get_variable('w', shape=(input_size, output_size), initializer=tf.contrib.layers.xavier_initializer())

        scale_b = tf.get_variable('b', shape=(output_size,), initializer=tf.zeros_initializer())

    return tf.matmul(input_tensor, scale_w) + scale_b
