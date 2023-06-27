import logging
import numpy as np
import math
from scipy import stats
from maxatac.utilities.system_tools import Mute
from copy import deepcopy

with Mute():
    import tensorflow as tf
    from tensorflow.keras import backend as K
    from tensorflow.keras.callbacks import ModelCheckpoint
    from tensorflow.keras.layers import (
        Input,
        Conv1D,
        MaxPooling1D,
        AveragePooling1D,
        Lambda,
        BatchNormalization,
        Dense,
        Flatten,
    )
    from tensorflow.keras import initializers
    from tensorflow.keras.activations import relu, gelu
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam

    #from maxatac.utilities.constants import INPUT_LENGTH, \
    #    OUTPUT_LENGTH, ADAM_BETA_1, ADAM_BETA_2, DEFAULT_ADAM_LEARNING_RATE, \
    #    DEFAULT_ADAM_DECAY, DM_DROPOUT_RATE, \
    #    CONV_TOWER_CROSSATT_CONFIGS_FUSION, DOWNSAMPLE_METHOD_CONV_TOWER_CROSSATT, USE_TOKEN, \
    #    WHOLE_ATTENTION_KWARGS_SELFATT_GENOME, NUM_MHA_SELFATT, EMBEDDING_SIZE_SELFATT, \
    #    WHOLE_ATTENTION_KWARGS_CROSSATT_SIGNAL, NUM_MHA_CROSSATT, EMBEDDING_SIZE_CROSSATT

    from maxatac.utilities.constants import KERNEL_INITIALIZER, INPUT_LENGTH, INPUT_CHANNELS, INPUT_FILTERS, \
        INPUT_KERNEL_SIZE, INPUT_ACTIVATION, OUTPUT_FILTERS, OUTPUT_KERNEL_SIZE, FILTERS_SCALING_FACTOR, DILATION_RATE, \
        OUTPUT_LENGTH, CONV_BLOCKS, PADDING, POOL_SIZE, ADAM_BETA_1, ADAM_BETA_2, DEFAULT_ADAM_LEARNING_RATE, \
        DEFAULT_ADAM_DECAY

    from maxatac.architectures.dcnn import loss_function, dice_coef, get_layer
    from maxatac.architectures.attention_module_TF import TransformerBlock, TransformerBlockCrossAtt

class Token(tf.keras.layers.Layer):
    """Append a class token to an input layer."""

    def build(self, input_shape):
        cls_init = tf.zeros_initializer()
        self.hidden_size = input_shape[-1]
        self.cls = tf.Variable(
            name="repr",
            initial_value=cls_init(shape=(1, 1, self.hidden_size), dtype="float32"),
            trainable=True,
        )

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        cls_broadcasted = tf.cast(
            tf.broadcast_to(self.cls, [batch_size, 1, self.hidden_size]),
            dtype=inputs.dtype,
        )
        return tf.concat([cls_broadcasted, inputs], 1)
    


def get_conv_block(
    inbound_layer, conv_block_config, base_name
):
    """
    Feed the input through some conv layers.
    This function is very similar to Tareian's get_layer function, with just some slight modification
    """
    for l in range(conv_block_config["num_layer"]):
        if conv_block_config["activation"] == "gelu":
            #TODO
            pass
            #activation = gelu()
        else:
            activation = tf.keras.layers.ReLU(name=base_name + f"_relu_{l+1}")
        inbound_layer = Conv1D(
            filters=conv_block_config["num_filters"],
            kernel_size=conv_block_config["kernel"],
            padding=conv_block_config["padding"],
            strides=conv_block_config["stride"],
            activation=activation,
            name=base_name + f"_conv_layer_{l+1}"
        )(inbound_layer)
        inbound_layer = BatchNormalization(name=base_name + f"_batch_norm_{l+1}")(inbound_layer)
        inbound_layer = activation(inbound_layer)

    return inbound_layer


def get_conv_tower(
    inbound_layer, conv_tower_configs, downsample_method, base_name
):
    """
    Feed the input through the tower of conv layers
    Input has shape (batch_size, 1024, 16)
    """
    count = 0
    for conv_block_config in conv_tower_configs:
        count += 1
        print(f"before conv block: {inbound_layer.shape}")
        inbound_layer = get_conv_block(
            inbound_layer, 
            conv_block_config,
            base_name=f"{base_name}_conv_tower_block_{count}"
        )
        print(f"after conv block: {inbound_layer.shape}")
        # After each conv block, use maxpooling to reduce seq len by 2
        # set option to downsample whether with maxpooling or conv1d stride 2
        if downsample_method == "maxpooling":
            inbound_layer = MaxPooling1D(pool_size=5, strides=2, padding="same", name=f"{base_name}_Conv_tower_block_{count}_maxpool")(inbound_layer)
        elif downsample_method == "conv":
            inbound_layer = Conv1D(filters=conv_block_config["num_filters"], kernel_size=conv_block_config["kernel"], strides=2, padding="same", name=f"{base_name}_Conv_tower_block_{count}_downsampling_conv")(inbound_layer)
        else:
            inbound_layer = inbound_layer
    
    return inbound_layer


def get_positional_encoding(
    inbound_layer, seq_len, depth, n=10000
):
    """
    Return a positional encoding for the transformer,
    Input is the matrix I of shape (None, seq_len, embed_size)
    The function generates an pos encoding matrix P with shape (None, seq_len, embed_size)
    The output is I + P
    """
    # Get a pos encoding layer (taken from https://www.tensorflow.org/text/tutorials/transformer#the_embedding_and_positional_encoding_layer)
    depth = depth/2

    positions = np.arange(seq_len)[:, np.newaxis]     # (seq, 1)
    depths = np.arange(depth)[np.newaxis, :]/depth   # (1, depth)

    angle_rates = 1 / (n**depths)         # (1, depth)
    angle_rads = positions * angle_rates      # (pos, depth)

    pos_encoding = np.concatenate(
        [np.sin(angle_rads), np.cos(angle_rads)],
        axis=-1)

    pos_encoding = tf.cast(pos_encoding, dtype=tf.float32)
    # Use Add layer to add the name to the layer
    inbound_layer = tf.keras.layers.Add(name="Add_positional_encoding")([inbound_layer, pos_encoding[tf.newaxis, :seq_len, :]])
    #inbound_layer = inbound_layer + pos_encoding[tf.newaxis, :seq_len, :]

    return inbound_layer

def get_multihead_attention(
    inbound_layer, key_dim, num_heads
):
    """
    Get a multi head attention module
    """
    # Check correct dimensions
    embed_dim = inbound_layer.shape[-1]
    assert embed_dim == key_dim*num_heads, "Embedding size has to be equal to key_dim * num heads"
    mha, att_scores = tf.keras.layers.MultiHeadAttention(key_dim=key_dim, num_heads=num_heads)(query=inbound_layer, value=inbound_layer, return_attention_scores=True)
    layer_norm = tf.keras.layers.LayerNormalization()(mha)
    inbound_layer = tf.keras.layers.Add()([inbound_layer, layer_norm])

    return inbound_layer, att_scores

def get_feed_forward_nn(
    inbound_layer, d_ff, base_name, activation_function="relu", bias1=True, bias2=True, 
):
    """
    Get the feed forward neural net right after multihead attention
    inbound_layer has shape (batch_size, seq_len, embed_dim)
    Implementation taken from https://nn.labml.ai/transformers/feed_forward.html
    From the original paper, the output from FFNN is also layer_normed and add residual
    """
    if activation_function == "relu":
        activation = tf.keras.layers.ReLU(name=base_name + "_relu")
    
    embed_dim = inbound_layer.shape[-1]
    dense_1 = tf.keras.layers.Dense(d_ff, use_bias=bias1, name=base_name + "_dense_1")
    dense_2 = tf.keras.layers.Dense(embed_dim, use_bias=bias2, name=base_name + "_dense_2")
    layer_norm = tf.keras.layers.LayerNormalization(name=base_name + "_layernorm_in_ffnn")
    residual = tf.keras.layers.Add(name=base_name + "_residual_in_ffnn")

    ffnn_output = activation(dense_1(inbound_layer))
    ffnn_output = dense_2(ffnn_output)
    inbound_layer = residual([inbound_layer, layer_norm(ffnn_output)])
    #inbound_layer = inbound_layer + layer_norm(ffnn_output)

    return inbound_layer


def get_multihead_attention_custom(
    inbound_layer, key_dim, num_heads, seq_len, base_name
):
    """
    Make a custom multi head attention layer
    inbound_layer has shape (batch_size, seq_len, embed_dim)
    Implementation taken from https://nn.labml.ai/transformers/mha.html
    """
    embed_dim = inbound_layer.shape[-1]
    assert embed_dim == key_dim*num_heads, "Embedding size has to be equal to key_dim * num heads"
    query = tf.keras.layers.Dense(embed_dim, name=base_name + "_Wq")
    key = tf.keras.layers.Dense(embed_dim, name=base_name + "_Wk")
    value = tf.keras.layers.Dense(embed_dim, name=base_name + "_Wv")

    wq = query(inbound_layer)
    wk = key(inbound_layer)
    wv = value(inbound_layer)
    # Reshape this from (batch_size, seq_len, embed_dim) to (batch_size, num_heads, seq_len, key_dim)
    wq = tf.reshape(wq, (-1, num_heads, seq_len, key_dim))
    wk = tf.reshape(wk, (-1, num_heads, seq_len, key_dim))
    wv = tf.reshape(wv, (-1, num_heads, seq_len, key_dim))

    # Take the dot product of wq and wk.T to produce (batch_size, num_heads, seq_len, seq_len)
    att_weights = tf.linalg.matmul(wq, tf.transpose(wk, perm=[0, 1, 3, 2]))
    att_weights = att_weights * (1/math.sqrt(key_dim))
    att_weights = tf.keras.layers.Softmax(axis=1, name=base_name + "_softmax_att_weights")(att_weights)

    # Multiply this with value to produce (batch_size, num_heads, seq_len, key_dim)
    output = tf.linalg.matmul(att_weights, wv)

    # Reshape to (batch_size, seq_len, embed_dim) and pass to another dense layer
    output = tf.reshape(output, (-1, seq_len, embed_dim))
    output = tf.keras.layers.Dense(embed_dim, name=base_name + "_dense_after_mha")(output)

    # Pass through layer norm and add residual
    layer_norm = tf.keras.layers.LayerNormalization(name=base_name + "_layernorm_in_mha")(output)
    output = tf.keras.layers.Add(name=base_name + "_residual_in_mha")([inbound_layer, layer_norm])
    #inbound_layer = inbound_layer + layer_norm

    return output, att_weights

def get_multiinput_crossatt_transformer(
        output_activation,
        model_config,
        adam_learning_rate=DEFAULT_ADAM_LEARNING_RATE,
        adam_decay=DEFAULT_ADAM_DECAY,
        input_length=INPUT_LENGTH,
        output_length=OUTPUT_LENGTH,
        adam_beta_1=ADAM_BETA_1,
        adam_beta_2=ADAM_BETA_2,
        weights=None
):
    """
    If weights are provided they will be loaded into created model
    """
    logging.debug("Building cross attention model")

    # Current there are two inputs: one for the genome sequence, one for the ATAC-seq signal
    genome_input = Input(shape=(input_length, 4), name="genome")
    signal_input = Input(shape=(input_length, 1), name="signal")

    # The current feature dim to the transformer is 64
    # Using 2 inputs, each input will be transformed to feature dim of 32
    # Then they are concatenated and passed through another conv layer to keep the same 64
    genome_layer = genome_input
    signal_layer = signal_input

    # Get the conv tower for each branch
    genome_layer = get_conv_tower(genome_layer, model_config["CONV_TOWER_CONFIGS_FUSION"]["genome"], model_config["DOWNSAMPLE_METHOD_CONV_TOWER"], "genome")
    signal_layer = get_conv_tower(signal_layer, model_config["CONV_TOWER_CONFIGS_FUSION"]["signal"], model_config["DOWNSAMPLE_METHOD_CONV_TOWER"], "signal")

    # Add the representation token
    if model_config["USE_TOKEN"]:
        genome_layer = Token(name="Add_representation_token_genome")(genome_layer)
        signal_layer = Token(name="Add_representation_token_signal")(signal_layer)

    # Pass the genome branch through the self-attention
    new_rpe_crossatt_genome_attention_kwargs = deepcopy(model_config["WHOLE_ATTENTION_KWARGS_SELFATT_GENOME"])
    new_rpe_crossatt_genome_attention_kwargs["initializer"] = initializers.get(model_config["WHOLE_ATTENTION_KWARGS_SELFATT_GENOME"]["initializer"])
    for i in range(model_config["NUM_MHA_SELFATT"]):
        transformer_block_selfatt = TransformerBlock(
            channels=model_config["EMBEDDING_SIZE_SELFATT"],
            dropout_rate=model_config["DM_DROPOUT_RATE"],
            attention_kwargs=new_rpe_crossatt_genome_attention_kwargs,
            name=f"Transformer_block_selfatt{i}"
        )
        outputs = transformer_block_selfatt(genome_layer)
        logging.error(f"Length of transformer block output: {len(outputs)}")
        genome_layer = outputs[0]
        

    # Pass the genome branch to the atacseq branch through cross attention
    new_rpe_crossatt_signal_attention_kwargs = deepcopy(model_config["WHOLE_ATTENTION_KWARGS_CROSSATT_SIGNAL"])
    new_rpe_crossatt_signal_attention_kwargs["initializer"] = initializers.get(model_config["WHOLE_ATTENTION_KWARGS_CROSSATT_SIGNAL"]["initializer"])
    for i in range(model_config["NUM_MHA_CROSSATT"]):
        transformer_block_crossatt = TransformerBlockCrossAtt(
            channels=model_config["EMBEDDING_SIZE_CROSSATT"],
            dropout_rate=model_config["DM_DROPOUT_RATE"],
            attention_kwargs=new_rpe_crossatt_signal_attention_kwargs,
            name=f"Transformer_block_crossatt{i}"
        )
        genome_layer, att_weights = transformer_block_crossatt(signal_layer, genome_layer)

    # Get the token from the sequences
    # signal_layer and genome_layer now has shape (None, seq_len, feature_dim)
    genome_token = tf.keras.layers.Lambda(lambda x: x[:, 0, :], name="Extract_signal_token")(genome_layer)  #shape = (None, feature_dim)

    # Pass through an MLP for final classification
    output_layer = tf.keras.layers.Dense(units=output_length, activation=output_activation, kernel_initializer='glorot_uniform')(genome_token)

    logging.debug("Added outputs layer: " + "\n - " + str(output_layer))

    # Model
    model = Model(inputs=[genome_input, signal_input], outputs=output_layer)

    model.compile(
        optimizer=Adam(
            lr=adam_learning_rate,
            beta_1=adam_beta_1,
            beta_2=adam_beta_2,
            weight_decay=adam_decay
        ),
        loss=tf.keras.losses.Poisson(),
        metrics=[dice_coef]
    )

    logging.debug("Model compiled")

    if weights is not None:
        model.load_weights(weights)
        logging.debug("Weights loaded")

    return model