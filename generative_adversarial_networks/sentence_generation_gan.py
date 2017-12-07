"""2017 - Script for a GAN that generates sentences."""

import tensorflow as tf
import arcadian.gm

class SentenceGenerationGAN(arcadian.gm.GenericModel):
    def __init__(self, code_size, **kwargs):
        self.code_size = code_size

        # Run setup of model
        super().__init__(**kwargs)

    def build(self):
        """Implement Tensorflow model and specify model placeholders and output tensors you wish to evaluate
        using self.input_placeholders and self.output_tensors dictionaries. Specify each entry as a name:tensor pair.
        Specify variable scopes to restore by adding to self.load_scopes list. Specify loss function to train on
        by assigning loss tensor to self.loss variable. Read initialize_loss() documentation for adaptive
        learning rates and evaluating loss tensor at runtime."""

        # Define inputs to model

        # Build generator

        # Build descriminator

        # Define loss

    def _define_inputs(self):
        """Create """


def build_resnet(name, tf_input, num_layers):
    """Transform input tensor by running it through ResNet layers.
    All layers are the same size as the input. All layers are initialized
    with a normal gaussian with mean=0 stdev=0.1.

    Arguments:
        - name: internal name used to create weights and biases (required)
        - tf_input: input tensor to enter ResNet
        - num_layers: number of ResNet layers

    Returns: Output result of ResNet layers.
    """

    # The size of resnet layers are the same (without rescaling layers).
    input_size = tf.shape(input)[1]

    # Each resnet layer consists of an input linear layer, a relu, and an output linear layer.
    # Now we build each layer of the beautiful resnet.
    with tf.variable_scope(name):
        for layer_index in range(num_layers):
            with tf.variable_scope('RESNET_LAYER_' + str(layer_index)):
                # Initialize input and output weights and biases
                tf_layer_input_weights = tf.get_variable('resnet_input_weights',
                                                   shape=(input_size, input_size),
                                                   initializer=tf.random_normal_initializer(stddev=0.1))
                tf_layer_input_biases = tf.get_variable('resnet_input_biases',
                                                  shape=(input_size),
                                                  initializer=tf.random_normal_initializer(stddev=0.1))
                tf_layer_output_weights = tf.get_variable('resnet_output_weights',
                                                   shape=(input_size, input_size),
                                                   initializer=tf.random_normal_initializer(stddev=0.1))
                tf_layer_output_biases = tf.get_variable('resnet_output_biases',
                                                  shape=(input_size),
                                                  initializer=tf.random_normal_initializer(stddev=0.1))
                # Perform transformation of input to produce output
                tf_input_linear_transform = tf.matmul(input, tf_layer_input_weights) + tf_layer_input_biases
                tf_cutoff = tf.nn.relu(tf_input_linear_transform)
                tf_output_linear_transform = tf.matmul(tf_cutoff, tf_layer_output_weights) + tf_layer_output_biases
                # Add output to input
                tf_input = tf_input + tf_output_linear_transform

    return tf_input
