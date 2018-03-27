"""2017 - Script for a GAN that maps from one representation to another. Uses Improved Wasserstein GAN implementation.
GAN uses ResNet architecture for both generator and discriminator. Each residual layer uses a
linear-relu-linear layer structure. No batch norm. The GAN was first used to generate latent representations of
sentences."""

import tensorflow as T
import arcadian.gm
import arcadian.dataset
import numpy as np

class ResNetGAN(arcadian.gm.GenericModel):
    def __init__(self, z_size, num_gen_layers, num_dsc_layers, num_dsc_trains, out_size, context_size=0, **kwargs):
        """This GAN network allows for a mapping from one representation to another using a ResNet architecture.
        Generator receives z vector and context vector as input, discriminator receives real or fake data vector
        and context vector. Generator produces samples, discriminator produces score per sample (Improved
        Wasserstein Implementation).

        z_size          - size of z input to generator (does not include context size)
        num_gen_layers  - number of resnet layers in generator
        num_dsc_layers  - number of discriminator layers in generator
        num_dsc_trains  - number of discriminator trains per generator train
        out_size        - size of generator output
        context_size    - size of context vector given to both generator and discriminator (Default: context vector
                          not included)"""

        self.z_size = z_size
        self.out_size = out_size
        self.context_size = context_size
        self.num_gen_layers = num_gen_layers
        self.num_dsc_layers = num_dsc_layers
        self.num_dsc_trains = num_dsc_trains

        self.dsc_scope = None

        # Run setup of model
        super().__init__(**kwargs)

    def build(self):
        """Override of Generic Model"""

        # Define inputs to model
        self._define_inputs()

        # Build generator
        with T.variable_scope('GENERATOR'):
            # Add context to generator input if it exists
            gen_input = self.z
            if self.context_size != 0:
                gen_input = T.concat([gen_input, self.context], axis=1)

            self.outputs = self._build_generator(gen_input, self.out_size)

            assert self.outputs.get_shape()[1] == self.out_size

        # Build descriminator
        with T.variable_scope('DISCRIMINATOR') as scope:

            self.dsc_scope = scope
            # Build two discriminators with same weights, one takes in generator output as input, the other
            # takes in true data samples.

            # Give context as input to discriminator if it exists
            dsc_input = self.outputs
            if self.context_size != 0:
                dsc_input = T.concat([dsc_input, self.context], axis=1)

            self.fake_scores = self._build_discriminator(dsc_input)

            assert len(self.fake_scores.get_shape()) == 1

            scope.reuse_variables()

            # Give context as input to discriminator if it exists
            dsc_input = self.data
            if self.context_size != 0:
                dsc_input = T.concat([dsc_input, self.context], axis=1)

            self.real_scores = self._build_discriminator(dsc_input)

            assert len(self.real_scores.get_shape()) == 1

        # Define loss
        with T.variable_scope('TRAINER'):
            norm_loss, grad_norm, dsc_real_loss, dsc_fake_loss, dsc_loss, gen_loss = self._build_loss()

        self.load_scopes = ['GENERATOR', 'DISCRIMINATOR']

        if self.context_size != 0:
            self.i.update({'context': self.context})

        # define interface
        self.i.update({'z': self.z, 'data': self.data,
                       'gen_lr': self.gen_lr, 'dsc_lr': self.dsc_lr})

        # out: code, fake_logits, real_logits, gradient_norm, dsc_real_loss, dsc_fake_loss, gen_loss
        self.o.update({'outputs': self.outputs, 'fake_scores': self.fake_scores, 'real_scores': self.real_scores,
                       'dsc_real_loss': dsc_real_loss, 'dsc_fake_loss': dsc_fake_loss, 'dsc_loss': dsc_loss,
                       'gen_loss': gen_loss, 'norm_loss': norm_loss, 'grad_norm': grad_norm})

    def _define_inputs(self):
        """Create latent space input"""

        # True label input to discriminator.
        self.data = T.placeholder(T.float32, shape=(None, self.out_size), name='code')

        # Only include context vector if size is not zero
        if self.context_size != 0:
            self.context = T.placeholder(T.float32, shape=(None, self.context_size), name='context')

        # Random vector z input to generator.
        self.z = T.placeholder(T.float32, shape=(None, self.z_size), name='z')

        self.gen_lr = T.placeholder_with_default(.001, ())  # learning rates
        self.dsc_lr = T.placeholder_with_default(.001, ())

    def _build_generator(self, inputs, output_size):
        """Build ResNet generator mapping from input vector z to
        a generated data example."""
        return build_resnet('generator', inputs, output_size, self.num_gen_layers)

    def _build_discriminator(self, data):
        """Build Discriminator using ResNet, with a linear layer and sigmoid
        to produce a binary classification of whether the input is a real
        or fake example.
        """

        resnet_layer = build_resnet('input_stage', data, int(data.get_shape()[1]), self.num_dsc_layers)
        decision_layer = build_linear_layer('decision_layer', resnet_layer, 1)
        logits = T.reshape(decision_layer, [-1])

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
        batch_size = T.shape(self.data)[0]
        real_data = self.data
        fake_data = self.outputs
        LAMBDA = 10

        alpha = T.random_uniform(
            shape=[batch_size, 1],
            minval=0.,
            maxval=1.
        )
        differences = fake_data - real_data
        interpolates = real_data + (alpha * differences)

        # add context to interpolates if it exists
        if self.context_size != 0:
            interpolates = T.concat([interpolates, self.context], axis=1)

        with T.variable_scope(self.dsc_scope) as scope:
            scope.reuse_variables()
            intpol_logits = self._build_discriminator(interpolates)

        gradients = T.gradients(intpol_logits, [interpolates])[0]
        slopes = T.sqrt(T.reduce_sum(T.square(gradients), reduction_indices=[1]))
        gradient_penalty = T.reduce_mean((slopes - 1.) ** 2)
        norm_loss = LAMBDA * gradient_penalty

        grad_norm = T.reduce_mean(slopes)

        # Wasserstein losses
        dsc_real_loss = -T.reduce_mean(self.real_scores)
        dsc_fake_loss = T.reduce_mean(self.fake_scores)

        dsc_loss = dsc_fake_loss + dsc_real_loss + norm_loss
        gen_loss = -dsc_fake_loss

        # Create optimizers for generator and discriminator.
        gen_op = T.train.AdamOptimizer(self.gen_lr).minimize(gen_loss, var_list=generator_variables)
        dsc_op = T.train.AdamOptimizer(self.dsc_lr).minimize(dsc_loss, var_list=discriminator_variables)

        # this tells the model to execute gen_op once followed by self.num_dsc_trains executions of the dsc_op
        self.train_ops = [gen_op] + [dsc_op] * self.num_dsc_trains

        return norm_loss, grad_norm, dsc_real_loss, dsc_fake_loss, dsc_loss, gen_loss

    def action_per_epoch(self, output_tensor_dict, epoch_index, is_training, params, **kwargs):
        """Optional: Define action to take place at the end of every epoch. Can use this
        for printing accuracy, saving statistics, etc. Remember, if is_training=False, we are using the model for
        prediction. Check for this. Returns true to continue training. Only return false if you wish to
        implement early-stopping."""
        if is_training:
            print('Showing per-epoch statistics')
            #print('Generator epoch loss: %s' % np.mean(output_tensor_dict['gen_loss']))
            print('Discriminator epoch loss: %s' % np.mean(output_tensor_dict['dsc_loss']))
            print()
            print('######################')
            print()

        return True

    def action_per_batch(self, input_batch_dict, output_batch_dict, epoch_index, batch_index, is_training,
                         params, **kwargs):
        """Optional: Define action to take place at the end of every batch. Can use this
        for printing accuracy, saving statistics, etc. Remember, if is_training=False, we are using the model for
        prediction. Check for this."""
        if batch_index % 1000 == 0 and is_training:
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
            print('Real Output: %s' % output_batch_dict['real_scores'][:10])
            print('Real Output Mean: %s' % np.mean(output_batch_dict['real_scores']))

            print('Fake Output: %s' % output_batch_dict['fake_scores'][:10])
            print('Fake Output Mean: %s' % np.mean(output_batch_dict['fake_scores']))

            print('Generator code: %s' % output_batch_dict['outputs'][:2, :10])
            print('Generator code mean: %s' % np.mean(output_batch_dict['outputs']))
            print('Generator code var: %s' % np.var(output_batch_dict['outputs']))

            print('Data mean: %s' % np.mean(input_batch_dict['data']))
            print('Data var: %s' % np.var(input_batch_dict['data']))

            print('Gradient norm: %s' % np.mean(output_batch_dict['grad_norm']))
            print()

        # Save every 1000 batches!
        if batch_index != 0 and batch_index % 1000 == 0 and is_training:
            print('Saving...')
            if self.save_per_epoch and self.trainable and is_training:
                self.saver.save(self.sess, self.save_dir, global_step=epoch_index)



    def action_before_training(self, placeholder_dict, num_epochs, is_training, output_tensor_names,
                               params, batch_size=32, train_op_names=None, **kwargs):
        """Optional: Define action to take place at the beginning of training/prediction, once. This could be
        used to set output_tensor_names so that certain ops always execute, as needed for other action functions."""
        if is_training:
            output_tensor_names.extend(['dsc_loss', 'gen_loss', 'real_scores', 'fake_scores', 'outputs', 'grad_norm'])




def build_linear_layer(name, input_tensor, output_size, xavier=False):
    """Build linear layer by creating random weight matrix and bias vector,
    and applying them to input. Weights initialized with random normal
    initializer.

    Arguments:
        - name: Required for unique Variable names
        - input: (num_examples x layer_size) matrix input
        - output_size: size of output Tensor

    Returns: Output Tensor of linear layer with size (num_examples, out_size).
        """

    if xavier:
        initializer = T.contrib.layers.xavier_initializer(uniform=False)
    else:
        initializer = T.random_normal_initializer(stddev=0.01)

    input_size = input_tensor.get_shape()[-1]  #tf.shape(input_tensor)[1]
    with T.variable_scope(name):
        scale_w = T.get_variable('w', shape=(input_size, output_size),
                                 initializer= initializer) # tf.contrib.layers.xavier_initializer(uniform=False)) #

        scale_b = T.get_variable('b', shape=(output_size,), initializer=T.zeros_initializer())

    return T.matmul(input_tensor, scale_w) + scale_b


def build_resnet(name, input_tensor, output_size, num_layers):
    """Transform input tensor by running it through ResNet layers.
    Allows for different input and output sizes. If they are different,
    the first half of the resnet layers have the size of the input.
    The second half of the resnet layers have the size of the output.
    A linear layer in the middle maps from the input size to the
    output size.


    Arguments:
        - name: internal name used to create weights and biases (required)
        - input: input tensor to enter ResNet
        - num_layers: number of ResNet layers

    Returns: Output result of ResNet layers.
    """

    # We want to vary the structure when using batch norm
    # normal structure: linear-relu-linear
    # with batch norm: BN-relu-linear-BN-relu-linear
    # so we add BN-relu at the beginning and BN before relu

    # The size of resnet layers are the same (without rescaling layers).
    input_size = int(input_tensor.get_shape()[1])

    # Each resnet layer consists of an input linear layer, a relu, and an output linear layer.
    # Now we build each layer of the beautiful resnet.
    with T.variable_scope(name):
        for layer_index in range(num_layers):
            with T.variable_scope('RESNET_LAYER_' + str(layer_index)):
                # Construct layer.
                prev_size = int(input_tensor.get_shape()[1])  # size of previous layer

                input_layer = build_linear_layer('input_layer', input_tensor, prev_size)
                tanh_layer = T.nn.relu(input_layer)
                output_layer = build_linear_layer('output_layer', tanh_layer, prev_size)

                # Add output to input.
                input_tensor = input_tensor + output_layer

                # add scaling layer from input size to output size in the middle of the resnet
                if input_size != output_size and layer_index == num_layers // 2:
                    input_tensor = build_linear_layer('scale_layer', input_tensor, output_size)

    return input_tensor


def get_variables_of_scope(scope):
    """Get all trainable variables under scope."""
    return T.get_collection(T.GraphKeys.TRAINABLE_VARIABLES, scope=scope)


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

class GaussianInterpolationDataset(arcadian.dataset.Dataset):
    def __init__(self, length, num_features, feature_name, scale=1.0):
        """This dataset picks two random Gaussian points, and constructs
        a linear path between these points. Each example in the dataset
        moves from the first to the second point for increasing index.

        Arguments:
            length - number of points to sample between start and end points
            num_features - dimensionality of Gaussian points
            feature_name - what feature name to use at interface of model
            scale - smaller scale means points are closer
            """
        self.length = length
        self.num_features = num_features
        self.feature_name = feature_name
        self.first_loc = np.random.randn(self.num_features)
        self.second_loc = np.random.randn(self.num_features)
        self.scale = scale

    def __getitem__(self, index):

        # p goes from the first point to the second point
        # for increasing index.
        # for scale less than one it never reaches second point
        a = index / (self.length-1) * self.scale

        p = self.first_loc + (self.second_loc - self.first_loc) * a

        return {self.feature_name: p}

    def __len__(self):
        return self.length