"""2017 - Script for a GAN that generates sentences."""

import tensorflow as tf
import arcadian.gm
import arcadian.dataset
import numpy as np

class SentenceGenerationGAN(arcadian.gm.GenericModel):
    def __init__(self, code_size, num_gen_layers, num_dsc_layers, num_dsc_trains, **kwargs):
        self.code_size = code_size
        self.num_gen_layers = num_gen_layers
        self.num_dsc_layers = num_dsc_layers
        self.num_dsc_trains = num_dsc_trains  # number of times to train discriminator per generator train

        self.dsc_scope = None

        # Run setup of model
        super().__init__(**kwargs)

    def build(self):
        """Implement Tensorflow model and specify model placeholders and output tensors you wish to evaluate
        using self.input_placeholders and self.output_tensors dictionaries. Specify each entry as a name:tensor pair.
        Specify variable scopes to restore by adding to self.load_scopes list. Specify loss function to train on
        by assigning loss tensor to self.loss variable. Read initialize_loss() documentation for adaptive
        learning rates and evaluating loss tensor at runtime."""

        # Notice: self.inputs['code'] is label sample, while self.outputs['code'] is generated sample

        # Define inputs to model
        self._define_inputs()

        # Build generator
        with tf.variable_scope('GENERATOR'):
            self.outputs['code'] = self._build_generator(self.inputs['z'])

            assert self.outputs['code'].get_shape()[1] == self.code_size

        # Build descriminator
        with tf.variable_scope('DISCRIMINATOR') as scope:

            self.dsc_scope = scope
            # Build two discriminators, one takes in generator output as input, the other
            # takes in true data samples.
            self.outputs['fake_logits'] = self._build_discriminator(self.outputs['code'])

            assert len(self.outputs['fake_logits'].get_shape()) == 1

            scope.reuse_variables()

            self.outputs['real_logits'] = self._build_discriminator(self.inputs['code'])

            assert len(self.outputs['real_logits'].get_shape()) == 1

        # Define loss
        with tf.variable_scope('TRAINER'):
            self._build_loss()

        self.load_scopes = ['GENERATOR', 'DISCRIMINATOR']

    def _define_inputs(self):
        """Create latent space input"""

        # True label input to discriminator.
        self.inputs['code'] = tf.placeholder(tf.float32, shape=(None, self.code_size), name='code')

        # Random vector z input to generator.
        self.inputs['z'] = tf.placeholder(tf.float32, shape=(None, self.code_size), name='z')

        self.inputs['gen_learning_rate'] = tf.placeholder_with_default(.001, ())
        self.inputs['dsc_learning_rate'] = tf.placeholder_with_default(.001, ())
        # Used in dropout
        self.inputs['keep_prob'] = tf.placeholder_with_default(1.0, (), name='keep_prob')
        # c value used for gradient clipping in Wasserstein GAN
        self.inputs['c'] = tf.placeholder(dtype=tf.float32, shape=(), name='c')

    def _build_generator(self, z):
        """Build ResNet generator mapping from input vector z to
        a generated data example."""
        return build_resnet('generator', z, self.num_gen_layers)

    def _build_discriminator(self, code):
        """Build Discriminator using ResNet, with a linear layer and sigmoid
        to produce a binary classification of whether the input is a real
        or fake example.

        Arguments:
            - code: Tensor input representing real or generated data example
        """

        resnet_layer = build_resnet('input_stage', code, self.num_dsc_layers)
        decision_layer = build_linear_layer('decision_layer', resnet_layer, 1)
        logits = tf.reshape(decision_layer, [-1])

        assert len(logits.get_shape()) == 1

        return logits

    def _build_loss(self):
        """Add optimizers to optimizer list for training both generator and
        discriminator."""
        generator_variables = get_variables_of_scope('GENERATOR')
        discriminator_variables = get_variables_of_scope('DISCRIMINATOR')

        assert len(generator_variables) > 0
        assert len(discriminator_variables) > 0

        # Norm clipping as suggested in 'Improved Training of Wasserstein GANs'
        batch_size = tf.shape(self.inputs['code'])[0]
        real_data = self.inputs['code']
        fake_data = self.outputs['code']
        LAMBDA = 10

        alpha = tf.random_uniform(
            shape=[batch_size, 1],
            minval=0.,
            maxval=1.
        )
        differences = fake_data - real_data
        interpolates = real_data + (alpha * differences)

        with tf.variable_scope(self.dsc_scope) as scope:
            scope.reuse_variables()
            intpol_logits = self._build_discriminator(interpolates)

        gradients = tf.gradients(intpol_logits, [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
        gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
        norm_loss = LAMBDA * gradient_penalty

        self.outputs['gradient_norm'] = tf.reduce_mean(slopes)

        # Wasserstein losses
        self.outputs['dsc_real_loss'] = -tf.reduce_mean(self.outputs['real_logits'])
        self.outputs['dsc_fake_loss'] = tf.reduce_mean(self.outputs['fake_logits'])

        self.outputs['dsc_loss'] = self.outputs['dsc_fake_loss'] + self.outputs['dsc_real_loss'] + norm_loss
        self.outputs['gen_loss'] = -self.outputs['dsc_fake_loss']

        gen_lr = self.inputs['gen_learning_rate']
        dsc_lr = self.inputs['dsc_learning_rate']


        # Create optimizers for generator and discriminator.
        gen_op = tf.train.AdamOptimizer(gen_lr).minimize(self.outputs['gen_loss'], var_list=generator_variables)
        dsc_op = tf.train.AdamOptimizer(dsc_lr).minimize(self.outputs['dsc_loss'], var_list=discriminator_variables)

        # Clip discriminator weights to be small
        #d_clip_op = tf.group(*[d.assign(tf.clip_by_value(d, -self.inputs['c'], self.inputs['c'])) for d in discriminator_variables])
        #g_clip_op = tf.group(*[g.assign(tf.clip_by_value(g, -0.01, 0.01)) for g in generator_variables])

        #gen_op = tf.group(gen_op, g_clip_op)
        #dsc_op = tf.group(dsc_op, d_clip_op)

        self.train_ops = [gen_op] + [dsc_op] * self.num_dsc_trains

    def action_per_epoch(self, output_tensor_dict, epoch_index, is_training, parameter_dict, **kwargs):
        """Optional: Define action to take place at the end of every epoch. Can use this
        for printing accuracy, saving statistics, etc. Remember, if is_training=False, we are using the model for
        prediction. Check for this. Returns true to continue training. Only return false if you wish to
        implement early-stopping."""
        print()
        print('######################')
        print()
        return True

    def action_per_batch(self, input_batch_dict, output_batch_dict, epoch_index, batch_index, is_training,
                         parameter_dict, **kwargs):
        """Optional: Define action to take place at the end of every batch. Can use this
        for printing accuracy, saving statistics, etc. Remember, if is_training=False, we are using the model for
        prediction. Check for this."""
        if batch_index % 100 == 0 and is_training:
            print()
            print('Generator Loss: %s' % output_batch_dict['gen_loss'])
            print('Discriminator Loss: %s' % output_batch_dict['dsc_loss'])

            # def step(x):
            #     return 1 * (x > 0)

            # generator_accuracy = np.mean(step(output_batch_dict['fake_logits']))
            # discriminator_fake_accuracy = 1 - generator_accuracy
            # discriminator_real_accuracy = np.mean(step(output_batch_dict['real_logits']))
            # discriminator_accuracy = (discriminator_fake_accuracy + discriminator_real_accuracy) / 2

            # print('Generator Accuracy: %s' % generator_accuracy)
            # print('Discriminator Accuracy: %s' % discriminator_accuracy)
            print('Real Output: %s' % output_batch_dict['real_logits'][:10])
            print('Real Output Mean: %s' % np.mean(output_batch_dict['real_logits']))

            print('Fake Output: %s' % output_batch_dict['fake_logits'][:10])
            print('Fake Output Mean: %s' % np.mean(output_batch_dict['fake_logits']))

            print('Generator code: %s' % output_batch_dict['code'][:2, :10])
            print('Generator code mean: %s' % np.mean(output_batch_dict['code']))
            print('Generator code var: %s' % np.var(output_batch_dict['code']))

            print('Data mean: %s' % np.mean(input_batch_dict['code']))
            print('Data var: %s' % np.var(input_batch_dict['code']))

            print('Gradient norm: %s' % np.mean(output_batch_dict['gradient_norm']))
            print()

        # Save every 1000 batches!
        if batch_index != 0 and batch_index % 1000 == 0 and is_training:
            print('Saving...')
            if self.save_per_epoch and self.trainable and is_training:
                self.saver.save(self.sess, self.save_dir, global_step=epoch_index)



    def action_before_training(self, placeholder_dict, num_epochs, is_training, output_tensor_names,
                               parameter_dict, batch_size=32, train_op_names=None, **kwargs):
        """Optional: Define action to take place at the beginning of training/prediction, once. This could be
        used to set output_tensor_names so that certain ops always execute, as needed for other action functions."""
        if is_training:
            output_tensor_names.extend(['dsc_loss', 'gen_loss', 'real_logits', 'fake_logits', 'code', 'gradient_norm'])




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
    input_size = input_tensor.get_shape()[1]  #tf.shape(input_tensor)[1]
    with tf.variable_scope(name):
        scale_w = tf.get_variable('w', shape=(input_size, output_size),
                                  initializer= tf.random_normal_initializer(stddev=0.01)) # tf.contrib.layers.xavier_initializer(uniform=False)) #

        scale_b = tf.get_variable('b', shape=(output_size,), initializer=tf.zeros_initializer())

    return tf.matmul(input_tensor, scale_w) + scale_b


def build_resnet(name, input_tensor, num_layers):
    """Transform input tensor by running it through ResNet layers.
    All layers are the same size as the input. All layers are initialized
    with xavier initialization, batch norm is applied.

    Arguments:
        - name: internal name used to create weights and biases (required)
        - input: input tensor to enter ResNet
        - num_layers: number of ResNet layers
        - is_training: only needed for batch norm, model is in training mode or test mode

    Returns: Output result of ResNet layers.
    """

    # We want to vary the structure when using batch norm
    # normal structure: linear-relu-linear
    # with batch norm: BN-relu-linear-BN-relu-linear
    # so we add BN-relu at the beginning and BN before relu

    # The size of resnet layers are the same (without rescaling layers).
    input_size = input_tensor.get_shape()[1]

    # Each resnet layer consists of an input linear layer, a relu, and an output linear layer.
    # Now we build each layer of the beautiful resnet.
    with tf.variable_scope(name):
        for layer_index in range(num_layers):
            with tf.variable_scope('RESNET_LAYER_' + str(layer_index)):
                # Construct layer.

                input_layer = build_linear_layer('input_layer', input_tensor, input_size)
                tanh_layer = tf.nn.relu(input_layer)
                output_layer = build_linear_layer('output_layer', tanh_layer, input_size)

                # Add output to input.
                input_tensor = input_tensor + output_layer

    return input_tensor


def get_variables_of_scope(scope):
    """Get all trainable variables under scope."""
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)


class GaussianRandomDataset(arcadian.dataset.Dataset):
    def __init__(self, length, num_features, feature_name):
        """Dataset that produces a random gaussian vector for every
        training example."""
        self.length = length
        self.num_features = num_features
        self.feature_name = feature_name

    def __getitem__(self, index):
        return {self.feature_name: np.random.randn(self.num_features)}

    def __len__(self):
        return self.length
