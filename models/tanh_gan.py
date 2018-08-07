from cic.models.rnet_gan import ResNetGAN
from cic.models.rnet_gan import build_linear_layer
import tensorflow as tf

class TanhResNetGAN(ResNetGAN):
    def _build_generator(self, inputs, output_size):
        """Here, we replace the ResNet layers of the
        original resnet gan with a new residual type. We run
        a linear transform on the input, apply a tanh and add
        back to the input at each time step. This is to mimick
        the LSTM, which adds the output of a tanh to the cell
        state at every timestep. We apply one final tanh at the
        output, to mimick the tanh output of the LSTM hidden state."""

        # The size of resnet layers are the same (without rescaling layers).
        input_size = int(inputs.get_shape()[1])

        # Each resnet layer consists of an input linear layer, a relu, and an output linear layer.
        # Now we build each layer of the beautiful resnet.
        with tf.variable_scope('TANH_RESNET'):
            for layer_index in range(self.num_gen_layers):
                with tf.variable_scope('RESNET_LAYER_' + str(layer_index)):
                    # Construct layer.
                    prev_size = int(inputs.get_shape()[1])  # size of previous layer

                    # Important! Here we just do a linear layer and a tanh
                    # We want output variance of residual to be at least equal to input variance,
                    # so we start with 1/prev_size.
                    # In practice, we want even smaller, such that the output variance
                    # of each layer + x is similar to the input variance
                    stddev = 1 / prev_size / 100
                    linear_layer = build_linear_layer('input_layer', inputs, prev_size, stddev=stddev)
                    tanh_layer = tf.nn.tanh(linear_layer)

                    # Add output to input.
                    inputs = inputs + tanh_layer

                    # add scaling layer from input size to output size in the middle of the resnet
                    if input_size != output_size and layer_index == self.num_gen_layers // 2:
                        inputs = build_linear_layer('scale_layer', inputs, output_size)

        # tranform output with tanh
        inputs = tf.nn.tanh(inputs)

        return inputs