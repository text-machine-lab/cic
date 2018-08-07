"""GAN which uses LSTM as generator and ResNet as discriminator."""
from cic.models.rnet_gan import ResNetGAN
import tensorflow as tf

class LSTMResNetGAN(ResNetGAN):
    def _build_generator(self, inputs, output_size):
        """Replace ResNet generator of base class GAN
        with an LSTM. The LSTM takes in input and processes
        one LSTM step at a time. Unlike normal LSTMs, which
        use the same weights for every time step but take in
        different inputs, this LSTM takes in the same input
        but uses different weights. This allows for a fixed
        size input space while allowing the generator to learn
        a complex function.

        Returns: The final output step of the LSTM."""

        batch_size = tf.shape(inputs)[0]
        zeros = tf.zeros([batch_size, output_size])
        state = (zeros, zeros)
        output = None
        for index in range(self.num_gen_layers):
            with tf.variable_scope('LSTM_STEP_' + str(index)):
                cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=output_size)
                output, state = cell(inputs, state)

        return output