import datetime

import tensorflow as tf

from open_seq2seq.data import Speech2TextDataLayer
from open_seq2seq.decoders import ListenAttendSpellDecoder
from open_seq2seq.encoders import ListenAttendSpellConvEncoder
from open_seq2seq.encoders.las_encoder import ListenAttendSpellStackEncoder
from open_seq2seq.losses import BasicSequenceLoss, CrossEntropyWithSmoothing
from open_seq2seq.models import Speech2Text
from open_seq2seq.optimizers.lr_policies import poly_decay, fixed_lr

base_model = Speech2Text

base_params = {
    "random_seed": 0,
    "use_horovod": False,   # True
    "num_epochs": 400,

    "num_gpus": 2,  # 8
    "batch_size_per_gpu": 32,  # 32  # 64
    "iter_size": 4,

    "save_summaries_steps": 1100,
    "print_loss_steps": 10,
    "print_samples_steps": 200,
    "eval_steps": 1100,
    "save_checkpoint_steps": 1100,
    "logdir": "experiments/las_librispeech/small_frame_stack/{}".format(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')),

    "optimizer": "Adam",
    "optimizer_params": {
    },
    "lr_policy": fixed_lr,
    "lr_policy_params": {
        "learning_rate": 1e-3
    },

    "max_grad_norm": 5.0,

    "regularizer": tf.contrib.layers.l2_regularizer,
    "regularizer_params": {
        'scale': 0.0001
    },

    # "dtype": "mixed",
    # "loss_scaling": "Backoff",

    "dtype": tf.float32,

    "summaries": ['learning_rate', 'variables', 'gradients', 'larc_summaries',
                  'variable_norm', 'gradient_norm', 'global_gradient_norm'],

    # "initializer": tf.contrib.layers.xavier_initializer,

    "encoder": ListenAttendSpellStackEncoder,
    "encoder_params": {

        "stack_num_frames": 4,

        "recurrent_layers": [
            {
                "type": "cudnn_lstm", "num_layers": 3,
                "hidden_dim": 256, "dropout_keep_prob": 1,
                "pool": False, "pool_size": [0], "stride": [0],
            }
        ],

        "dropout_keep_prob": 0.8,

        "initializer": tf.contrib.layers.xavier_initializer,
        "initializer_params": {
            'uniform': False,
        },
        "activation_fn": tf.nn.relu,
        "data_format": "channels_last",
    },

    "decoder": ListenAttendSpellDecoder,
    "decoder_params": {
        "tgt_emb_size": 256,
        "pos_embedding": True,

        "attention_params": {
            "attention_dim": 256,
            "attention_type": "chorowski",
            "use_coverage": True,
            "num_heads": 1,
            "plot_attention": True,
        },

        "rnn_type": "lstm",
        "hidden_dim": 512,
        "num_layers": 1,

        "dropout_keep_prob": 0.8,

        "sampling_probability": 0.0,

        "beam_width": 4,
        "use_language_model": False,
    },
    "loss": BasicSequenceLoss,
    "loss_params": {
        "offset_target_by_one": False,
        "average_across_timestep": True,
        "do_mask": True,
    },
}

train_params = {
    "data_layer": Speech2TextDataLayer,
    "data_layer_params": {
        "num_audio_features": 80,
        "input_type": "logfbank",
        "augmentation": {
        },
        # "bpe": True,
        "vocab_file": "open_seq2seq/test_utils/toy_speech_data/vocab.txt",
        "dataset_files": [
            "data/librispeech/librivox-train-clean-100.csv",
            "data/librispeech/librivox-train-clean-360.csv",
            "data/librispeech/librivox-train-other-500.csv",
        ],
        "max_duration": 16.7,
        "shuffle": True,
        "autoregressive": True,
    },
}

eval_params = {
    "data_layer": Speech2TextDataLayer,
    "data_layer_params": {
        "num_audio_features": 80,
        "input_type": "logfbank",
        "vocab_file": "open_seq2seq/test_utils/toy_speech_data/vocab.txt",
        "dataset_files": [
            "data/librispeech/librivox-dev-clean.csv",
        ],
        "shuffle": False,
        "autoregressive": True,
    },
}

infer_params = {
    "data_layer": Speech2TextDataLayer,
    "data_layer_params": {
        "num_audio_features": 80,
        "input_type": "logfbank",
        "vocab_file": "open_seq2seq/test_utils/toy_speech_data/vocab.txt",
        "dataset_files": [
            "data/librispeech/librivox-test-clean.csv",
        ],
        "shuffle": False,
        "autoregressive": True,
    },
}
