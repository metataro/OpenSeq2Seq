# Copyright (c) 2018 NVIDIA Corporation
from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals

import math

import tensorflow as tf
from tensorflow.contrib.cudnn_rnn.python.layers.cudnn_rnn import CUDNN_RNN_BIDIRECTION

from .encoder import Encoder
from open_seq2seq.parts.cnns.conv_blocks import conv_actv, conv_bn_actv

cells_dict = {
    "lstm": tf.nn.rnn_cell.BasicLSTMCell,
    "gru": tf.nn.rnn_cell.GRUCell,
}

def rnn_layer(layer_type, num_layers, name, inputs, src_length, hidden_dim,
              regularizer, training, dropout_keep_prob=1.0):
  """Helper function that applies convolution and activation.
    Args:
      layer_type: the following types are supported
        'lstm', 'gru'
  """
  rnn_cell = cells_dict[layer_type]
  dropout = tf.nn.rnn_cell.DropoutWrapper

  multirnn_cell_fw = tf.nn.rnn_cell.MultiRNNCell(
      [dropout(rnn_cell(hidden_dim),
               output_keep_prob=dropout_keep_prob)
       for _ in range(num_layers)]
  )

  multirnn_cell_bw = tf.nn.rnn_cell.MultiRNNCell(
      [dropout(rnn_cell(hidden_dim),
               output_keep_prob=dropout_keep_prob)
       for _ in range(num_layers)]
  )

  output, state = tf.nn.bidirectional_dynamic_rnn(
      cell_fw=multirnn_cell_fw, cell_bw=multirnn_cell_bw,
      inputs=inputs,
      sequence_length=src_length,
      dtype=inputs.dtype,
      scope=name,
  )
  output = tf.concat(output, 2)

  return output


class ListenAttendSpellEncoder(Encoder):
  """Listen Attend Spell like encoder with support for reduction in time dimension of the input.
  Can use convolutional layers, recurrent layers or both.
  """

  @staticmethod
  def get_required_params():
    return dict(Encoder.get_required_params(), **{
        'dropout_keep_prob': float,
        'recurrent_layers': list,
        'convnet_layers': list,
        'activation_fn': None,  # any valid callable
    })

  @staticmethod
  def get_optional_params():
    return dict(Encoder.get_optional_params(), **{
        'data_format': ['channels_first', 'channels_last'],
        'normalization': [None, 'batch_norm'],
        'bn_momentum': float,
        'bn_epsilon': float,
    })

  def __init__(self, params, model, name="las_encoder", mode='train'):
    """Listen Attend Spell like encoder constructor.

    See parent class for arguments description.

    Config parameters:

    * **dropout_keep_prop** (float) --- keep probability for dropout.
    * **convnet_layers** (list) --- list with the description of convolutional
      layers. For example::
        "convnet_layers": [
          {
            "type": "conv1d", "repeat" : 5,
            "kernel_size": [7], "stride": [1],
            "num_channels": 250, "padding": "SAME"
          },
          {
            "type": "conv1d", "repeat" : 1,
            "kernel_size": [1], "stride": [2],
            "num_channels": 1000, "padding": "SAME"
          },
        ]
    * **recurrent_layers** (list) --- list with the description of recurrent
      layers. For example::
        "recurrent_layers": [
            {
                "type": "lstm", "num_layers": 1,
                "hidden_dim": 512, "dropout_keep_prob": 0.8,
                "pool": True, "pool_size":[2], "stride": [2],
            },
            {
                "type": "lstm", "num_layers": 3,
                "hidden_dim": 512, "dropout_keep_prob": 0.8,
                "pool": False, "pool_size":[-1], "stride": [-1],
            },
        ], 
    * **activation_fn** --- activation function to use.
    * **data_format** (string) --- could be either "channels_first" or
      "channels_last". Defaults to "channels_last".
    * **normalization** --- normalization to use. Accepts [None, 'batch_norm'].
      Use None if you don't want to use normalization. Defaults to 'batch_norm'.     
    * **bn_momentum** (float) --- momentum for batch norm. Defaults to 0.90.
    * **bn_epsilon** (float) --- epsilon for batch norm. Defaults to 1e-3.
    """
    super(ListenAttendSpellEncoder, self).__init__(params, model, name, mode)

  def _encode(self, input_dict):
    """Creates TensorFlow graph for Wav2Letter like encoder.

    Args:
      input_dict (dict): input dictionary that has to contain
          the following fields::
            input_dict = {
              "source_tensors": [
                src_sequence (shape=[batch_size, sequence length, num features]),
                src_length (shape=[batch_size])
              ]
            }

    Returns:
      dict: dictionary with the following tensors::

        {
          'outputs': hidden state, shape=[batch_size, sequence length, n_hidden]
          'src_length': tensor, shape=[batch_size]
        }
    """

    source_sequence, src_length = input_dict['source_tensors']

    training = (self._mode == "train")
    dropout_keep_prob = self.params['dropout_keep_prob'] if training else 1.0
    regularizer = self.params.get('regularizer', None)
    normalization = self.params.get('normalization', 'batch_norm')
    data_format = self.params.get('data_format', 'channels_last')

    normalization_params = {}
    if normalization is None:
      conv_block = conv_actv
    elif normalization == "batch_norm":
      conv_block = conv_bn_actv
      normalization_params['bn_momentum'] = self.params.get(
          'bn_momentum', 0.90)
      normalization_params['bn_epsilon'] = self.params.get('bn_epsilon', 1e-3)
    else:
      raise ValueError("Incorrect normalization")

    conv_feats = source_sequence

    # ----- Convolutional layers ---------------------------------------------
    convnet_layers = self.params['convnet_layers']

    for idx_convnet in range(len(convnet_layers)):
      layer_type = convnet_layers[idx_convnet]['type']
      layer_repeat = convnet_layers[idx_convnet]['repeat']
      ch_out = convnet_layers[idx_convnet]['num_channels']
      kernel_size = convnet_layers[idx_convnet]['kernel_size']
      strides = convnet_layers[idx_convnet]['stride']
      padding = convnet_layers[idx_convnet]['padding']
      dropout_keep = convnet_layers[idx_convnet].get(
          'dropout_keep_prob', dropout_keep_prob) if training else 1.0

      if padding == "VALID":
        src_length = (src_length - kernel_size[0]) // strides[0] + 1
      else:
        src_length = (src_length + strides[0] - 1) // strides[0]

      for idx_layer in range(layer_repeat):
        conv_feats = conv_block(
            layer_type=layer_type,
            name="conv{}{}".format(
                idx_convnet + 1, idx_layer + 1),
            inputs=conv_feats,
            filters=ch_out,
            kernel_size=kernel_size,
            activation_fn=self.params['activation_fn'],
            strides=strides,
            padding=padding,
            regularizer=regularizer,
            training=training,
            data_format=data_format,
            **normalization_params
        )
        conv_feats = tf.nn.dropout(x=conv_feats, keep_prob=dropout_keep)

    rnn_feats = conv_feats
    rnn_block = rnn_layer

    # ----- Recurrent layers ---------------------------------------------
    recurrent_layers = self.params['recurrent_layers']

    for idx_rnn in range(len(recurrent_layers)):
      layer_type = recurrent_layers[idx_rnn]['type']
      num_layers = recurrent_layers[idx_rnn]['num_layers']
      hidden_dim = recurrent_layers[idx_rnn]['hidden_dim']
      dropout_keep = recurrent_layers[idx_rnn].get(
          'dropout_keep_prob', dropout_keep_prob) if training else 1.0
      use_pool = recurrent_layers[idx_rnn]['pool']
      pool_size = recurrent_layers[idx_rnn]['pool_size']
      strides = recurrent_layers[idx_rnn]['stride']

      rnn_feats = rnn_block(
          layer_type=layer_type,
          num_layers=num_layers,
          name="rnn{}".format(
              idx_rnn + 1),
          inputs=rnn_feats,
          src_length=src_length,
          hidden_dim=hidden_dim,
          regularizer=regularizer,
          training=training,
          dropout_keep_prob=dropout_keep,
      )

      if use_pool:
        rnn_feats = tf.layers.max_pooling1d(
          inputs=rnn_feats,
          pool_size=pool_size,
          strides=strides,
        )
        src_length = (src_length - pool_size[0]) // strides[0] + 1
    outputs = rnn_feats

    return {
        'outputs': outputs,
        'src_length': src_length,
    }


class ListenAttendSpellConvEncoder(Encoder):
    """Listen Attend Spell like encoder with support for reduction in time dimension of the input.
    Can use convolutional layers, recurrent layers or both.
    """

    @staticmethod
    def get_required_params():
        return dict(Encoder.get_required_params(), **{
            'dropout_keep_prob': float,
            'recurrent_layers': list,
            'convnet_layers': list,
            'activation_fn': None,  # any valid callable
        })

    @staticmethod
    def get_optional_params():
        return dict(Encoder.get_optional_params(), **{
            'data_format': ['channels_first', 'channels_last'],
            'normalization': [None, 'batch_norm'],
            'bn_momentum': float,
            'bn_epsilon': float,
        })

    def __init__(self, params, model, name="las_encoder", mode='train'):
        """Listen Attend Spell like encoder constructor.

        See parent class for arguments description.

        Config parameters:

        * **dropout_keep_prop** (float) --- keep probability for dropout.
        * **convnet_layers** (list) --- list with the description of convolutional
          layers. For example::
            "convnet_layers": [
              {
                "type": "conv1d", "repeat" : 5,
                "kernel_size": [7], "stride": [1],
                "num_channels": 250, "padding": "SAME"
              },
              {
                "type": "conv1d", "repeat" : 1,
                "kernel_size": [1], "stride": [2],
                "num_channels": 1000, "padding": "SAME"
              },
            ]
        * **recurrent_layers** (list) --- list with the description of recurrent
          layers. For example::
            "recurrent_layers": [
                {
                    "type": "lstm", "num_layers": 1,
                    "hidden_dim": 512, "dropout_keep_prob": 0.8,
                    "pool": True, "pool_size":[2], "stride": [2],
                },
                {
                    "type": "lstm", "num_layers": 3,
                    "hidden_dim": 512, "dropout_keep_prob": 0.8,
                    "pool": False, "pool_size":[-1], "stride": [-1],
                },
            ],
        * **activation_fn** --- activation function to use.
        * **data_format** (string) --- could be either "channels_first" or
          "channels_last". Defaults to "channels_last".
        * **normalization** --- normalization to use. Accepts [None, 'batch_norm'].
          Use None if you don't want to use normalization. Defaults to 'batch_norm'.
        * **bn_momentum** (float) --- momentum for batch norm. Defaults to 0.90.
        * **bn_epsilon** (float) --- epsilon for batch norm. Defaults to 1e-3.
        """
        super(ListenAttendSpellConvEncoder, self).__init__(params, model, name, mode)

    def _encode(self, input_dict):
        """Creates TensorFlow graph for Wav2Letter like encoder.

        Args:
          input_dict (dict): input dictionary that has to contain
              the following fields::
                input_dict = {
                  "source_tensors": [
                    src_sequence (shape=[batch_size, sequence length, num features]),
                    src_length (shape=[batch_size])
                  ]
                }

        Returns:
          dict: dictionary with the following tensors::

            {
              'outputs': hidden state, shape=[batch_size, sequence length, n_hidden]
              'src_length': tensor, shape=[batch_size]
            }
        """

        source_sequence, src_length = input_dict['source_tensors']

        training = (self._mode == "train")
        dropout_keep_prob = self.params['dropout_keep_prob'] if training else 1.0
        regularizer = self.params.get('regularizer', None)
        normalization = self.params.get('normalization', 'batch_norm')
        data_format = self.params.get('data_format', 'channels_last')

        normalization_params = {}
        if normalization is None:
            conv_block = conv_actv
        elif normalization == "batch_norm":
            conv_block = conv_bn_actv
            normalization_params['bn_momentum'] = self.params.get(
                'bn_momentum', 0.90)
            normalization_params['bn_epsilon'] = self.params.get('bn_epsilon', 1e-3)
        else:
            raise ValueError("Incorrect normalization")

        input_layer = tf.expand_dims(source_sequence, axis=-1)  # BTFC

        batch_size = input_layer.get_shape().as_list()[0]
        freq = input_layer.get_shape().as_list()[2]

        conv_feats = input_layer

        # ----- Convolutional layers ---------------------------------------------
        convnet_layers = self.params['convnet_layers']

        for idx_convnet, convnet_layer in enumerate(convnet_layers):
            layer_type = convnet_layer['type']
            layer_repeat = convnet_layer['repeat']
            ch_out = convnet_layer['num_channels']
            kernel_size = convnet_layer['kernel_size']
            strides = convnet_layer['stride']
            padding = convnet_layer['padding']
            dropout_keep = convnet_layer.get('dropout_keep_prob', dropout_keep_prob) if training else 1.0
            use_pool = convnet_layer['pool']
            pool_size = convnet_layer['pool_size']
            pool_strides = convnet_layer['pool_stride']

            for idx_layer in range(layer_repeat):
                conv_feats = conv_block(
                    layer_type=layer_type,
                    name="conv{}{}".format(idx_convnet + 1, idx_layer + 1),
                    inputs=conv_feats,
                    filters=ch_out,
                    kernel_size=kernel_size,
                    activation_fn=self.params['activation_fn'],
                    strides=strides,
                    padding=padding,
                    regularizer=regularizer,
                    training=training,
                    data_format=data_format,
                    **normalization_params
                )
                conv_feats = tf.nn.dropout(x=conv_feats, keep_prob=dropout_keep)

                if padding == "VALID":
                    src_length = (src_length - kernel_size[0]) // strides[0] + 1
                else:
                    src_length = (src_length + strides[0] - 1) // strides[0]

                if use_pool:
                    conv_feats = tf.layers.max_pooling2d(
                        inputs=conv_feats,
                        pool_size=pool_size,
                        strides=pool_strides,
                    )
                    src_length = (src_length - pool_size[0]) // pool_strides[0] + 1

        f = conv_feats.get_shape().as_list()[2]
        c = conv_feats.get_shape().as_list()[3]
        fc = f * c
        conv_feats = tf.reshape(conv_feats, [batch_size, -1, fc])

        conv_feats = tf.Print(conv_feats, [tf.shape(conv_feats)], message='shape_conv_feats=', summarize=100)

        rnn_feats = conv_feats

        # ----- Recurrent layers ---------------------------------------------
        recurrent_layers = self.params['recurrent_layers']

        for idx_rnn in range(len(recurrent_layers)):
            layer_type = recurrent_layers[idx_rnn]['type']
            num_layers = recurrent_layers[idx_rnn]['num_layers']
            hidden_dim = recurrent_layers[idx_rnn]['hidden_dim']
            dropout_keep = recurrent_layers[idx_rnn].get('dropout_keep_prob', dropout_keep_prob) if training else 1.0
            use_pool = recurrent_layers[idx_rnn]['pool']
            pool_size = recurrent_layers[idx_rnn]['pool_size']
            strides = recurrent_layers[idx_rnn]['stride']

            if layer_type == 'cudnn_lstm':
                rnn_block = tf.contrib.cudnn_rnn.CudnnLSTM(
                    num_layers=num_layers,
                    num_units=hidden_dim,
                    direction=CUDNN_RNN_BIDIRECTION,
                    dropout=1.0 - dropout_keep_prob,
                    dtype=rnn_feats.dtype,
                    name="cudnn_lstm",
                )
                rnn_feats, state = rnn_block(rnn_feats)
                # rnn_feats = tf.transpose(rnn_feats, [1, 0, 2])
            else:
                rnn_feats = rnn_layer(
                    layer_type=layer_type,
                    num_layers=num_layers,
                    name="rnn{}".format(
                        idx_rnn + 1),
                    inputs=rnn_feats,
                    src_length=src_length,
                    hidden_dim=hidden_dim,
                    regularizer=regularizer,
                    training=training,
                    dropout_keep_prob=dropout_keep,
                )

            if use_pool:
                rnn_feats = tf.layers.max_pooling1d(
                    inputs=rnn_feats,
                    pool_size=pool_size,
                    strides=strides,
                )
                src_length = (src_length - pool_size[0]) // strides[0] + 1
        outputs = rnn_feats

        outputs = tf.Print(outputs, [src_length], message='src_length=', summarize=100)
        outputs = tf.Print(outputs, [tf.shape(outputs)], message='shape_outputs=', summarize=100)

        return {
            'outputs': outputs,
            'src_length': src_length,
        }


class ListenAttendSpellStackEncoder(Encoder):
    """Listen Attend Spell like encoder with support for reduction in time dimension of the input.
    Can use convolutional layers, recurrent layers or both.
    """

    @staticmethod
    def get_required_params():
        return dict(Encoder.get_required_params(), **{
            'dropout_keep_prob': float,
            'recurrent_layers': list,
            'stack_num_frames': int,
            'activation_fn': None,  # any valid callable
        })

    @staticmethod
    def get_optional_params():
        return dict(Encoder.get_optional_params(), **{
            'data_format': ['channels_first', 'channels_last'],
            'bn_momentum': float,
            'bn_epsilon': float,
        })

    def __init__(self, params, model, name="las_encoder", mode='train'):
        """Listen Attend Spell like encoder constructor.

        See parent class for arguments description.

        Config parameters:

        * **dropout_keep_prop** (float) --- keep probability for dropout.
        * **convnet_layers** (list) --- list with the description of convolutional
          layers. For example::
            "convnet_layers": [
              {
                "type": "conv1d", "repeat" : 5,
                "kernel_size": [7], "stride": [1],
                "num_channels": 250, "padding": "SAME"
              },
              {
                "type": "conv1d", "repeat" : 1,
                "kernel_size": [1], "stride": [2],
                "num_channels": 1000, "padding": "SAME"
              },
            ]
        * **recurrent_layers** (list) --- list with the description of recurrent
          layers. For example::
            "recurrent_layers": [
                {
                    "type": "lstm", "num_layers": 1,
                    "hidden_dim": 512, "dropout_keep_prob": 0.8,
                    "pool": True, "pool_size":[2], "stride": [2],
                },
                {
                    "type": "lstm", "num_layers": 3,
                    "hidden_dim": 512, "dropout_keep_prob": 0.8,
                    "pool": False, "pool_size":[-1], "stride": [-1],
                },
            ],
        * **activation_fn** --- activation function to use.
        * **data_format** (string) --- could be either "channels_first" or
          "channels_last". Defaults to "channels_last".
        * **bn_momentum** (float) --- momentum for batch norm. Defaults to 0.90.
        * **bn_epsilon** (float) --- epsilon for batch norm. Defaults to 1e-3.
        """
        super(ListenAttendSpellStackEncoder, self).__init__(params, model, name, mode)

    def _encode(self, input_dict):
        """Creates TensorFlow graph for Wav2Letter like encoder.

        Args:
          input_dict (dict): input dictionary that has to contain
              the following fields::
                input_dict = {
                  "source_tensors": [
                    src_sequence (shape=[batch_size, sequence length, num features]),
                    src_length (shape=[batch_size])
                  ]
                }

        Returns:
          dict: dictionary with the following tensors::

            {
              'outputs': hidden state, shape=[batch_size, sequence length, n_hidden]
              'src_length': tensor, shape=[batch_size]
            }
        """

        source_sequence, src_length = input_dict['source_tensors']

        training = (self._mode == "train")
        dropout_keep_prob = self.params['dropout_keep_prob'] if training else 1.0
        regularizer = self.params.get('regularizer', None)
        data_format = self.params.get('data_format', 'channels_last')

        input_layer = tf.expand_dims(source_sequence, axis=-1)  # BTFC

        batch_size = input_layer.get_shape().as_list()[0]
        freq = input_layer.get_shape().as_list()[2]

        # ----- Frame stack layer ---------------------------------------------

        stack_num_frames = self.params['stack_num_frames']

        assert data_format == 'channels_last'

        stack_feats = input_layer

        sequence_length = tf.shape(stack_feats)[1]
        new_sequence_length = tf.ceil(tf.div(tf.to_float(sequence_length), float(stack_num_frames)))
        pad_sequence_length = tf.to_int32(new_sequence_length) * stack_num_frames - sequence_length

        stack_feats = tf.pad(stack_feats, [[0, 0], [0, pad_sequence_length], [0, 0], [0, 0]], 'CONSTANT', constant_values=0)
        stack_feats = tf.reshape(stack_feats, [batch_size, new_sequence_length, stack_num_frames*freq])

        src_length = tf.to_int32(tf.ceil(tf.div(tf.to_float(src_length), float(stack_num_frames))))

        rnn_feats = stack_feats

        # ----- Recurrent layers ---------------------------------------------
        recurrent_layers = self.params['recurrent_layers']

        for idx_rnn in range(len(recurrent_layers)):
            layer_type = recurrent_layers[idx_rnn]['type']
            num_layers = recurrent_layers[idx_rnn]['num_layers']
            hidden_dim = recurrent_layers[idx_rnn]['hidden_dim']
            dropout_keep = recurrent_layers[idx_rnn].get('dropout_keep_prob', dropout_keep_prob) if training else 1.0
            use_pool = recurrent_layers[idx_rnn]['pool']
            pool_size = recurrent_layers[idx_rnn]['pool_size']
            strides = recurrent_layers[idx_rnn]['stride']

            if layer_type == 'cudnn_lstm':
                rnn_block = tf.contrib.cudnn_rnn.CudnnLSTM(
                    num_layers=num_layers,
                    num_units=hidden_dim,
                    direction=CUDNN_RNN_BIDIRECTION,
                    dropout=1.0 - dropout_keep_prob,
                    dtype=rnn_feats.dtype,
                    name="cudnn_lstm",
                )
                rnn_feats, state = rnn_block(rnn_feats)
                # rnn_feats = tf.transpose(rnn_feats, [1, 0, 2])
            else:
                rnn_feats = rnn_layer(
                    layer_type=layer_type,
                    num_layers=num_layers,
                    name="rnn{}".format(
                        idx_rnn + 1),
                    inputs=rnn_feats,
                    src_length=src_length,
                    hidden_dim=hidden_dim,
                    regularizer=regularizer,
                    training=training,
                    dropout_keep_prob=dropout_keep,
                )

            if use_pool:
                rnn_feats = tf.layers.max_pooling1d(
                    inputs=rnn_feats,
                    pool_size=pool_size,
                    strides=strides,
                )
                src_length = (src_length - pool_size[0]) // strides[0] + 1
        outputs = rnn_feats
        print(outputs.get_shape())

        return {
            'outputs': outputs,
            'src_length': src_length,
        }
