import numpy as np
import tensorflow as tf

class WaveNet(object):
    def __init__(self,
                 dilations,
                 residual_channels,
                 dilation_channels,
                 skip_channels,
                 use_aux_features,
                 n_mfcc,
                 quantization_channels):

        self.dilations = dilations
        self.residual_channels = residual_channels
        self.dilation_channels = dilation_channels
        self.quantization_channels = quantization_channels
        self.skip_channels = skip_channels
        self.use_aux_features = use_aux_features
        self.n_mfcc = n_mfcc
        self.receptive_field = sum(self.dilations) + 2
        self.variables = self.create_variables()

    def create_variable(self, name, shape):
        initializer = tf.contrib.layers.xavier_initializer_conv2d()
        variable = tf.Variable(initializer(shape=shape), name=name)
        return variable

    def create_variables(self):
        ''' Create variables for the WaveNet'''

        variables = dict()
        with tf.variable_scope('wavenet'):
            with tf.variable_scope('causal_layer'):
                variables['causal_layer'] = self.create_variable('filter',
                    [2,self.quantization_channels, self.residual_channels])

            variables['dilated_stack'] = list()
            with tf.variable_scope('dilated_stack'):
                for i in range(len(self.dilations)):
                    with tf.variable_scope('layer{}'.format(i)):
                        current = dict()
                        current['filter'] = self.create_variable( 'filter',
                            [2, self.residual_channels, self.dilation_channels])
                        current['gate'] = self.create_variable('gate',
                            [2, self.residual_channels, self.dilation_channels])
                        current['dense'] = self.create_variable( 'dense',
                            [1, self.dilation_channels, self.residual_channels])
                        current['skip'] = self.create_variable('skip',
                            [1, self.dilation_channels, self.skip_channels])

                        if self.use_aux_features:
                            current['aux_gateweights'] = self.create_variable('aux_gate',
                                [1, self.n_mfcc, self.dilation_channels])
                            current['aux_filtweights'] = self.create_variable('aux_filter',
                                [1, self.n_mfcc, self.dilation_channels])

                        current['filter_bias'] = tf.get_variable('filter_bias',
                            [self.dilation_channels], initializer=tf.zeros_initializer)
                        current['gate_bias'] = tf.get_variable('gate_bias',
                            [self.dilation_channels], initializer=tf.zeros_initializer)
                        current['dense_bias'] = tf.get_variable('dense_bias',
                            [self.dilation_channels], initializer=tf.zeros_initializer)
                        current['skip_bias'] = tf.get_variable('skip_bias',
                            [self.skip_channels], initializer=tf.zeros_initializer)

                        variables['dilated_stack'].append(current)

            with tf.variable_scope('skip'):
                current = dict()
                current['conv1'] = self.create_variable('conv1',
                    [1, self.skip_channels, self.skip_channels])
                current['conv2'] = self.create_variable('conv2',
                    [1, self.skip_channels, self.quantization_channels])

                current['conv1_bias'] = tf.get_variable('conv1_bias',
                    [self.skip_channels], initializer=tf.zeros_initializer)
                current['conv2_bias'] = tf.get_variable('conv2_bias',
                    [self.quantization_channels], initializer=tf.zeros_initializer)

                variables['skip'] = current

        return variables

    def dilation_layer(self, input_batch, layer_index, dilation,
                                output_width, aux_input):

        variables = self.variables['dilated_stack'][layer_index]

        weights_filter = variables['filter']
        weights_gate = variables['gate']

        with tf.name_scope('causal_conv'):
            conv_filter = tf.nn.convolution(input_batch, weights_filter,
                                    padding='VALID', dilation_rate=[dilation])
            conv_gate = tf.nn.convolution(input_batch, weights_gate,
                                    padding='VALID', dilation_rate=[dilation])

        # trim aux input to make width of gate & filter
        # put aux input through 1x1 convolution and add to filter
        width = tf.shape(conv_gate)[1]
        aux_input = tf.slice(aux_input, [0, 0, 0], [-1, width, -1])
        weights_aux_filter = variables['aux_filtweights']
        conv_filter = conv_filter + tf.nn.conv1d(aux_input, weights_aux_filter,
                                                             stride=1,
                                                             padding="SAME",
                                                             name="aux_filter")
        # add bias
        filter_bias = variables['filter_bias']
        conv_filter = tf.add(conv_filter, filter_bias)

        # put aux input through 1x1 convolution and add to filter
        weights_aux_gate = variables['aux_gateweights']
        conv_gate = conv_gate + tf.nn.conv1d(aux_input, weights_aux_gate,
                                                             stride=1,
                                                             padding="SAME",
                                                             name="aux_gate")
        # add bias
        gate_bias = variables['gate_bias']
        conv_gate = tf.add(conv_gate, gate_bias)

        # activation functions
        out = tf.tanh(conv_filter) * tf.sigmoid(conv_gate)

        # The 1x1 conv to produce the residual output
        weights_dense = variables['dense']
        transformed = tf.nn.conv1d(
            out, weights_dense, stride=1, padding="SAME", name="dense")

        # The 1x1 conv to produce the skip output
        skip_cut = tf.shape(out)[1] - output_width
        out_skip = tf.slice(out, [0, skip_cut, 0], [-1, -1, -1])
        weights_skip = variables['skip']
        skip_contribution = tf.nn.conv1d(
            out_skip, weights_skip, stride=1, padding="SAME", name="skip")

        dense_bias = variables['dense_bias']
        skip_bias = variables['skip_bias']
        transformed = transformed + dense_bias
        skip_contribution = skip_contribution + skip_bias

        input_cut = tf.shape(input_batch)[1] - tf.shape(transformed)[1]
        input_batch = tf.slice(input_batch, [0, input_cut, 0], [-1, -1, -1])

        return skip_contribution, input_batch + transformed

    def create_network(self, input_batch, aux_input):

        # Initial convolution layer
        with tf.name_scope('causal_layer'):
            current_layer = tf.nn.conv1d(input_batch,
                                        self.variables['causal_layer'],
                                        stride=1,
                                        padding='VALID')

        output_width = tf.shape(input_batch)[1] - self.receptive_field + 1

        # Create dilation layers
        skip_outputs = []
        with tf.name_scope('dilated_stack'):
            for layer_index, dilation in enumerate(self.dilations):
                with tf.name_scope('layer{}'.format(layer_index)):
                    output, current_layer = self.dilation_layer(
                        current_layer, layer_index, dilation, output_width,
                        aux_input)
                    skip_outputs.append(output)

        with tf.name_scope('skip'):
            # process skip outputs
            summed = sum(skip_outputs)
            # 1st RELU and 1x1
            summed = tf.nn.relu(summed)
            summed = tf.nn.conv1d(summed,
                                self.variables['skip']['conv1'],
                                stride=1,
                                padding="SAME")
            summed = tf.add(summed, self.variables['skip']['conv1_bias'])
            # 2nd RELU and 1x1
            summed = tf.nn.relu(summed)
            summed = tf.nn.conv1d(summed,
                                self.variables['skip']['conv2'],
                                stride=1,
                                padding="SAME")
            summed = tf.add(summed, self.variables['skip']['conv2_bias'])

        return summed

    def generate(self, waveform, aux_input, use_aux_features):
        with tf.name_scope('wavenet'):
            encoded = tf.one_hot(waveform, self.quantization_channels, dtype=tf.float32)
            encoded = tf.reshape(encoded, [1, -1, self.quantization_channels])
            raw_output = self.create_network(encoded, aux_input)
            out = tf.reshape(raw_output, [-1, self.quantization_channels])
            proba = tf.nn.softmax(out)
            last = tf.slice(proba, [tf.shape(proba)[0] - 1, 0],
                [1, self.quantization_channels])
            return tf.reshape(last, [-1])

    def loss(self, input_batch, aux_input=None):
        ''' Creates a WaveNet and returns the loss '''
        with tf.name_scope('wavenet'):
            network_input_width = tf.shape(input_batch)[1] - 1
            network_input = tf.slice(input_batch, [0, 0, 0],
                                     [-1, network_input_width, -1])
            raw_output = self.create_network(network_input, aux_input)

            with tf.name_scope('loss'):
                target_output = tf.slice(
                    tf.reshape(
                        input_batch,
                        [1, -1, self.quantization_channels]),
                    [0, self.receptive_field, 0],
                    [-1, -1, -1])
                target_output = tf.reshape(target_output,
                                           [-1, self.quantization_channels])
                prediction = tf.reshape(raw_output,
                                        [-1, self.quantization_channels])
                loss = tf.nn.softmax_cross_entropy_with_logits_v2(
                    logits=prediction,
                    labels=target_output)
                reduced_loss = tf.reduce_mean(loss)

                tf.summary.scalar('loss', reduced_loss)

                return reduced_loss
