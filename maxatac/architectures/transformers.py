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

    from maxatac.utilities.constants import (
        KERNEL_INITIALIZER,
        INPUT_LENGTH,
        INPUT_CHANNELS,
        INPUT_FILTERS,
        INPUT_KERNEL_SIZE,
        INPUT_ACTIVATION,
        OUTPUT_FILTERS,
        OUTPUT_KERNEL_SIZE,
        FILTERS_SCALING_FACTOR,
        DILATION_RATE,
        OUTPUT_LENGTH,
        CONV_BLOCKS,
        PADDING,
        POOL_SIZE,
        ADAM_BETA_1,
        ADAM_BETA_2,
        DEFAULT_ADAM_LEARNING_RATE,
        DEFAULT_ADAM_DECAY,
        NUM_HEADS,
        NUM_MHA,
        KEY_DIMS,
        D_FF,
        CONV_TOWER_CONFIGS,
        EMBEDDING_SIZE,
        POOL_SIZE_BEFORE_FLATTEN,
        DOWNSAMPLE_METHOD_CONV_TOWER,
        INCEPTION_BRANCHES,
        WHOLE_ATTENTION_KWARGS,
        USE_RPE,
        DM_DROPOUT_RATE,
    )

    from maxatac.architectures.dcnn import loss_function, dice_coef, get_layer
    from maxatac.architectures.attention_module_TF import TransformerBlock


def get_inception_block(inbound_layer, inception_branches, base_name):
    """
    Get an inception block. The motivation is that, the motif can have multiple lengths, so using the inception block with multiple paths may capture this pattern
    Let's say the input has shape (batch, seq_len, embed_dim), after inception it is (batch, seq_len, embed_dim)
    The 4 filter sizes can be 7, 10, 13, and 16, with padding same and stride 1
    We can use 1x1 conv to reduce the embed dim like in the paper
    """
    outputs = []
    embed_dim = inbound_layer.shape[-1]
    assert (
        embed_dim % len(inception_branches) == 0
    ), "Embed dim has to be a multiple of number of branches"
    for i, branch in enumerate(inception_branches):
        # Each branch is a conv block with a conv + batch_norm + activation
        temp_layer = inbound_layer
        for j, block in enumerate(branch):
            if block["name"] == "conv":
                temp_layer = get_conv_block(
                    temp_layer, block, f"{base_name}_branch_{i}_block_{j}"
                )
            else:
                temp_layer = AveragePooling1D(
                    pool_size=block["pool_size"],
                    strides=block["stride"],
                    padding=block["padding"],
                    name=f"{base_name}_avgpool_branch_{i}_block_{j}",
                )(temp_layer)

        outputs.append(temp_layer)

    # Reshape the output into the desired shape
    output_layer = tf.keras.layers.Concatenate(axis=-1)(outputs)
    return output_layer


def get_conv_block(inbound_layer, conv_block_config, base_name):
    """
    Feed the input through some conv layers.
    This function is very similar to Tareian's get_layer function, with just some slight modification
    """
    for l in range(conv_block_config["num_layer"]):
        if conv_block_config["activation"] == "gelu":
            # TODO
            pass
            # activation = gelu()
        else:
            activation = tf.keras.layers.ReLU(name=base_name + f"_relu_{l+1}")
        inbound_layer = Conv1D(
            filters=conv_block_config["num_filters"],
            kernel_size=conv_block_config["kernel"],
            padding=conv_block_config["padding"],
            strides=conv_block_config["stride"],
            activation=activation,
            name=base_name + f"_conv_layer_{l+1}",
        )(inbound_layer)
        inbound_layer = BatchNormalization(name=base_name + f"_batch_norm_{l+1}")(
            inbound_layer
        )
        inbound_layer = activation(inbound_layer)

    return inbound_layer


def get_conv_tower(inbound_layer, conv_tower_configs, downsample_method):
    """
    Feed the input through the tower of conv layers
    Input has shape (batch_size, 1024, 16)
    """
    count = 0
    for conv_block_config in conv_tower_configs:
        count += 1
        print(f"before conv block: {inbound_layer.shape}")
        inbound_layer = get_conv_block(
            inbound_layer, conv_block_config, base_name=f"Conv_tower_block_{count}"
        )
        print(f"after conv block: {inbound_layer.shape}")
        # After each conv block, use maxpooling to reduce seq len by 2
        # set option to downsample whether with maxpooling or conv1d stride 2
        if downsample_method == "maxpooling":
            inbound_layer = MaxPooling1D(
                pool_size=5,
                strides=2,
                padding="same",
                name=f"Conv_tower_block_{count}_maxpool",
            )(inbound_layer)
        else:
            inbound_layer = Conv1D(
                filters=conv_block_config["num_filters"],
                kernel_size=conv_block_config["kernel"],
                strides=2,
                padding="same",
                name=f"Conv_tower_block_{count}_downsampling_conv",
            )(inbound_layer)

    return inbound_layer


def get_positional_encoding(inbound_layer, seq_len, depth, n=10000):
    """
    Return a positional encoding for the transformer,
    Input is the matrix I of shape (None, seq_len, embed_size)
    The function generates an pos encoding matrix P with shape (None, seq_len, embed_size)
    The output is I + P
    """
    # Get a pos encoding layer (taken from https://www.tensorflow.org/text/tutorials/transformer#the_embedding_and_positional_encoding_layer)
    depth = depth / 2

    positions = np.arange(seq_len)[:, np.newaxis]  # (seq, 1)
    depths = np.arange(depth)[np.newaxis, :] / depth  # (1, depth)

    angle_rates = 1 / (n**depths)  # (1, depth)
    angle_rads = positions * angle_rates  # (pos, depth)

    pos_encoding = np.concatenate([np.sin(angle_rads), np.cos(angle_rads)], axis=-1)

    pos_encoding = tf.cast(pos_encoding, dtype=tf.float32)
    # Use Add layer to add the name to the layer
    inbound_layer = tf.keras.layers.Add(name="Add_positional_encoding")(
        [inbound_layer, pos_encoding[tf.newaxis, :seq_len, :]]
    )
    # inbound_layer = inbound_layer + pos_encoding[tf.newaxis, :seq_len, :]

    return inbound_layer


def get_multihead_attention(inbound_layer, key_dim, num_heads):
    """
    Get a multi head attention module
    """
    # Check correct dimensions
    embed_dim = inbound_layer.shape[-1]
    assert (
        embed_dim == key_dim * num_heads
    ), "Embedding size has to be equal to key_dim * num heads"
    mha, att_scores = tf.keras.layers.MultiHeadAttention(
        key_dim=key_dim, num_heads=num_heads
    )(query=inbound_layer, value=inbound_layer, return_attention_scores=True)
    layer_norm = tf.keras.layers.LayerNormalization()(mha)
    inbound_layer = tf.keras.layers.Add()([inbound_layer, layer_norm])

    return inbound_layer, att_scores


def get_feed_forward_nn(
    inbound_layer,
    d_ff,
    base_name,
    activation_function="relu",
    bias1=True,
    bias2=True,
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
    dense_2 = tf.keras.layers.Dense(
        embed_dim, use_bias=bias2, name=base_name + "_dense_2"
    )
    layer_norm = tf.keras.layers.LayerNormalization(
        name=base_name + "_layernorm_in_ffnn"
    )
    residual = tf.keras.layers.Add(name=base_name + "_residual_in_ffnn")

    ffnn_output = activation(dense_1(inbound_layer))
    ffnn_output = dense_2(ffnn_output)
    inbound_layer = residual([inbound_layer, layer_norm(ffnn_output)])
    # inbound_layer = inbound_layer + layer_norm(ffnn_output)

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
    assert (
        embed_dim == key_dim * num_heads
    ), "Embedding size has to be equal to key_dim * num heads"
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
    att_weights = att_weights * (1 / math.sqrt(key_dim))
    att_weights = tf.keras.layers.Softmax(
        axis=1, name=base_name + "_softmax_att_weights"
    )(att_weights)

    # Multiply this with value to produce (batch_size, num_heads, seq_len, key_dim)
    output = tf.linalg.matmul(att_weights, wv)

    # Reshape to (batch_size, seq_len, embed_dim) and pass to another dense layer
    output = tf.reshape(output, (-1, seq_len, embed_dim))
    output = tf.keras.layers.Dense(embed_dim, name=base_name + "_dense_after_mha")(
        output
    )

    # Pass through layer norm and add residual
    layer_norm = tf.keras.layers.LayerNormalization(
        name=base_name + "_layernorm_in_mha"
    )(output)
    output = tf.keras.layers.Add(name=base_name + "_residual_in_mha")(
        [inbound_layer, layer_norm]
    )
    # inbound_layer = inbound_layer + layer_norm

    return output, att_weights


def get_transformer(
    output_activation,
    rpe_attention_kwargs=WHOLE_ATTENTION_KWARGS,
    use_rpe=USE_RPE,
    dm_dropout_rate=DM_DROPOUT_RATE,
    conv_tower_config=CONV_TOWER_CONFIGS,
    inception_block_config=INCEPTION_BRANCHES,
    mha_embedding_dim=EMBEDDING_SIZE,
    downsample_method_conv_tower=DOWNSAMPLE_METHOD_CONV_TOWER,
    num_mha=NUM_MHA,
    num_heads=NUM_HEADS,
    key_dim=KEY_DIMS,
    d_ff=D_FF,
    adam_learning_rate=DEFAULT_ADAM_LEARNING_RATE,
    adam_decay=DEFAULT_ADAM_DECAY,
    input_length=INPUT_LENGTH,
    input_channels=INPUT_CHANNELS,
    input_filters=INPUT_FILTERS,
    input_kernel_size=INPUT_KERNEL_SIZE,
    input_activation=INPUT_ACTIVATION,
    output_filters=OUTPUT_FILTERS,
    output_kernel_size=OUTPUT_KERNEL_SIZE,
    filters_scaling_factor=FILTERS_SCALING_FACTOR,
    dilation_rate=DILATION_RATE,
    output_length=OUTPUT_LENGTH,
    conv_blocks=CONV_BLOCKS,
    padding=PADDING,
    pool_size=POOL_SIZE,
    pool_size_before_flatten=POOL_SIZE_BEFORE_FLATTEN,
    adam_beta_1=ADAM_BETA_1,
    adam_beta_2=ADAM_BETA_2,
    target_scale_factor=1,
    dense_b=False,
    weights=None,
):
    """
    If weights are provided they will be loaded into created model
    """
    logging.debug("Building Dilated CNN model")

    # Inputs
    input_layer = Input(shape=(input_length, input_channels))

    # Temporary variables
    layer = input_layer  # redefined in encoder/decoder loops
    filters = input_filters  # redefined in encoder/decoder loops

    # Get the initial stem conv layer
    layer = get_layer(
        inbound_layer=layer,
        filters=filters,
        kernel_size=input_kernel_size,
        activation=input_activation,
        padding=padding,
        dilation_rate=1,
        kernel_initializer=KERNEL_INITIALIZER,
        n=1,
    )

    # Get the conv tower (output has shape batch_size, seq_len, embed_dim)
    layer = get_conv_tower(layer, conv_tower_config, downsample_method_conv_tower)

    # Add an inception block
    # layer = get_inception_block(layer, inception_block_config, "Inception_block")

    # get seq_len after the conv tower
    seq_len = layer.shape[1]

    if not use_rpe:
        # A positional encoding layer
        layer = get_positional_encoding(layer, seq_len=seq_len, depth=mha_embedding_dim)

        # Stack a list of encoders
        for i in range(num_mha):
            # The weights are in the shape of (batch_size, num_head, seq_len, seq_len)
            layer, att_weights = get_multihead_attention_custom(
                layer, key_dim, num_heads, seq_len, base_name=f"Encoder_{i}"
            )
            layer = get_feed_forward_nn(layer, d_ff, base_name=f"Encoder_{i}")

    else:
        new_rpe_attention_kwargs = deepcopy(rpe_attention_kwargs)
        new_rpe_attention_kwargs["initializer"] = initializers.get(
            rpe_attention_kwargs["initializer"]
        )
        for i in range(num_mha):
            deepmind_transformer_block = TransformerBlock(
                channels=mha_embedding_dim,
                dropout_rate=dm_dropout_rate,
                attention_kwargs=new_rpe_attention_kwargs,
                name=f"Transformer_block_{i}",
            )
            layer, att_weights = deepmind_transformer_block(layer)

    # Final postprocessing

    # Outputs
    layer_dilation_rate = dilation_rate[-1]
    if dense_b:
        output_layer = get_layer(
            inbound_layer=layer,
            filters=output_filters,
            kernel_size=output_kernel_size,
            activation=input_activation,
            padding=padding,
            dilation_rate=layer_dilation_rate,
            kernel_initializer=KERNEL_INITIALIZER,
            skip_batch_norm=True,
            n=1,
        )
    else:
        output_layer = get_layer(
            inbound_layer=layer,
            filters=output_filters,
            kernel_size=output_kernel_size,
            activation=output_activation,
            padding=padding,
            dilation_rate=layer_dilation_rate,
            kernel_initializer=KERNEL_INITIALIZER,
            skip_batch_norm=True,
            n=1,
        )

    # Downsampling from 1024 to 32 (change this) for a dynamic change
    seq_len = output_layer.shape[1]
    strides = ((seq_len - pool_size_before_flatten + 1) // output_length) + 1

    output_layer = tf.keras.layers.MaxPooling1D(
        pool_size=pool_size_before_flatten, strides=strides, padding="valid"
    )(output_layer)
    # Depending on the output activation functions, model outputs need to be scaled appropriately
    output_layer = Flatten()(output_layer)
    if dense_b:
        output_layer = Dense(
            output_length,
            activation=output_activation,
            kernel_initializer="glorot_uniform",
        )(output_layer)

    logging.debug("Added outputs layer: " + "\n - " + str(output_layer))

    # Model
    model = Model(inputs=[input_layer], outputs=output_layer)

    model.compile(
        optimizer=Adam(
            lr=adam_learning_rate,
            beta_1=adam_beta_1,
            beta_2=adam_beta_2,
            weight_decay=adam_decay,
        ),
        loss=loss_function,
        metrics=[dice_coef],
    )

    logging.debug("Model compiled")

    if weights is not None:
        model.load_weights(weights)
        logging.debug("Weights loaded")

    return model
