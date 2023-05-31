import tensorflow as tf
import numpy as np
import json
import os
import logging
import pandas as pd
import ntpath
import matplotlib.pyplot as plt
import seaborn as sns

from shap import GradientExplainer

from maxatac.utilities.training_tools import DataGenerator, get_input_matrix
from maxatac.utilities.genome_tools import load_bigwig, load_2bit
from maxatac.utilities.constants import BP_RESOLUTION

from maxatac.utilities.training_tools import MaxATACModel, ROIPool
from maxatac.architectures.dcnn import loss_function

def input_for_interfusion_ism(input_arr, interfusion):
    """
    Get the correct input for interfusion
    """
    assert len(input_arr.shape) == 3, "Shape of input is not 3"
    if interfusion:
        genome = input_arr[:, :, :4]
        atac = np.expand_dims(input_arr[:, :, -1], axis=-1)
        return {"genome": genome, "atac": atac}
    else:
        return input_arr
    
def input_for_interfusion_att(input_arr, interfusion):
    """
    Get the correct input for interfusion
    """
    assert len(input_arr.shape) == 3, "Shape of input is not 3"
    if interfusion:
        if input_arr.shape[0] == 1:
            genome = input_arr[0:1, :, :4]
            atac = np.expand_dims(input_arr[0:1, :, -1], axis=-1)
            return [{"genome": genome, "atac": atac}]
        input_arrs = []
        for s in range(input_arr.shape[0]):
            genome = input_arr[s:s+1, :, :4]
            atac = np.expand_dims(input_arr[s:s+1, :, -1], axis=-1)
            input_arrs.append({"genome": genome, "atac": atac})
        return input_arrs
    else:
        return [input_arr]

def string_to_char_array(seq):
    """
    Converts an ASCII string to a NumPy array of byte-long ASCII codes.
    e.g. "ACGT" becomes [65, 67, 71, 84].
    """
    return np.frombuffer(bytearray(seq, "utf8"), dtype=np.int8)

def char_array_to_string(arr):
    """
    Converts a NumPy array of byte-long ASCII codes into an ASCII string.
    e.g. [65, 67, 71, 84] becomes "ACGT".
    """
    return arr.tostring().decode("ascii")


def one_hot_to_tokens(one_hot):
    """
    Converts an L x D one-hot encoding into an L-vector of integers in the range
    [0, D], where the token D is used when the one-hot encoding is all 0. This
    assumes that the one-hot encoding is well-formed, with at most one 1 in each
    column (and 0s elsewhere).
    """
    tokens = np.tile(one_hot.shape[1], one_hot.shape[0])  # Vector of all D
    seq_inds, dim_inds = np.where(one_hot)
    tokens[seq_inds] = dim_inds
    return tokens


def tokens_to_one_hot(tokens, one_hot_dim):
    """
    Converts an L-vector of integers in the range [0, D] to an L x D one-hot
    encoding. The value `D` must be provided as `one_hot_dim`. A token of D
    means the one-hot encoding is all 0s.
    """
    identity = np.identity(one_hot_dim + 1)[:, :-1]  # Last row is all 0s
    return identity[tokens]

def dinuc_shuffle(seq, num_shufs=None, rng=None):
    """
    Creates shuffles of the given sequence, in which dinucleotide frequencies
    are preserved.
    Arguments:
        `seq`: either a string of length L, or an L x D NumPy array of one-hot
            encodings
        `num_shufs`: the number of shuffles to create, N; if unspecified, only
            one shuffle will be created
        `rng`: a NumPy RandomState object, to use for performing shuffles
    If `seq` is a string, returns a list of N strings of length L, each one
    being a shuffled version of `seq`. If `seq` is a 2D NumPy array, then the
    result is an N x L x D NumPy array of shuffled versions of `seq`, also
    one-hot encoded. If `num_shufs` is not specified, then the first dimension
    of N will not be present (i.e. a single string will be returned, or an L x D
    array).
    """
    if type(seq) is str:
        arr = string_to_char_array(seq)
    elif type(seq) is np.ndarray and len(seq.shape) == 2:
        seq_len, one_hot_dim = seq.shape
        arr = one_hot_to_tokens(seq)
    else:
        raise ValueError("Expected string or one-hot encoded array")

    if not rng:
        rng = np.random.RandomState()
   
    # Get the set of all characters, and a mapping of which positions have which
    # characters; use `tokens`, which are integer representations of the
    # original characters
    chars, tokens = np.unique(arr, return_inverse=True)

    # For each token, get a list of indices of all the tokens that come after it
    shuf_next_inds = []
    for t in range(len(chars)):
        mask = tokens[:-1] == t  # Excluding last char
        inds = np.where(mask)[0]
        shuf_next_inds.append(inds + 1)  # Add 1 for next token
 
    if type(seq) is str:
        all_results = []
    else:
        all_results = np.empty(
            (num_shufs if num_shufs else 1, seq_len, one_hot_dim),
            dtype=seq.dtype
        )

    for i in range(num_shufs if num_shufs else 1):
        # Shuffle the next indices
        for t in range(len(chars)):
            inds = np.arange(len(shuf_next_inds[t]))
            inds[:-1] = rng.permutation(len(inds) - 1)  # Keep last index same
            shuf_next_inds[t] = shuf_next_inds[t][inds]

        counters = [0] * len(chars)
       
        # Build the resulting array
        ind = 0
        result = np.empty_like(tokens)
        result[0] = tokens[ind]
        for j in range(1, len(tokens)):
            t = tokens[ind]
            ind = shuf_next_inds[t][counters[t]]
            counters[t] += 1
            result[j] = tokens[ind]

        if type(seq) is str:
            all_results.append(char_array_to_string(chars[result]))
        else:
            all_results[i] = tokens_to_one_hot(chars[result], one_hot_dim)
    return all_results if num_shufs else all_results[0]


def dinuc_shuffle_DNA_only_several_times(onehot_seq, ATAC_signals, num_samples, interfusion,
                                         seed=1234):
    """
    onehot_seq.shape = (1, seq_len, 4)
    ATAC_signals.shape = (1, seq_len, 1)
    """
    if len(onehot_seq.shape) == 3: onehot_seq = onehot_seq.reshape(-1, 4)
    if len(ATAC_signals.shape) == 1: ATAC_signals = ATAC_signals.reshape(-1, 1)
    rng = np.random.RandomState(seed)
    if not(interfusion):
        to_return = np.array([np.concatenate((dinuc_shuffle(onehot_seq, rng=rng),
                                            ATAC_signals), axis=1) for _ in range(num_samples)])
        return [to_return]
    
    else:
        genome = np.array([dinuc_shuffle(onehot_seq, rng=rng) for _ in range(num_samples)])
        signal = np.array([ATAC_signals for _ in range(num_samples)])
        return [genome, signal]
    

def background_atac_for_IG(genome_seq, ATAC_signal, num_samples, interfusion,
                            seed=1234):
    """
    Provide a background ATAC signal for IG
    genome_seq.shape = (1, seq_len, 4)
    ATAC_signal.shape = (1, seq_len, 1)
    """
    # For simplicity, choose the baseline to be a zero-vector
    # Then interpolate until reaching the original ATAC-seq signal
    alphas = np.linspace(0., 1., num_samples)
    atac_background = np.expand_dims((alphas*ATAC_signal[0]).T, axis=-1)    # shape = (num_samples, seq_len, 1)
    genome_background = np.repeat(genome_seq, repeats=num_samples, axis=0)
    
    if interfusion: return [genome_background, atac_background]


def load_model_from_dir(model_base_dir, model_config):
    """
    Load the model from dir
    """
    with open(os.path.join(model_base_dir, "best_epoch.txt"), "r") as f:
        best_model_dir = f.read().strip()

    with open(os.path.join(model_base_dir, "cmd_args.json"), "rb") as f:
        train_args = json.load(f)

    model = MaxATACModel(
        arch=train_args["arch"],
        model_config=model_config,
        seed=train_args["seed"],
        output_directory=train_args["output"],
        prefix=train_args["prefix"],
        threads=train_args["threads"],
        meta_path=train_args["meta_file"],
        output_activation=train_args["output_activation"],
        dense=train_args["dense"],
        weights=best_model_dir
    ).nn_model

    return model


def get_data(meta_file, chromosome, cell_type, output_dir, moods_bigwig=None):
    """
    Get data
    """
    sequence = "/users/ngun7t/opt/maxatac/data/hg38/hg38.2bit"
    df_dir = meta_file
    train_examples = ROIPool(chroms=[chromosome],
                            roi_file_path=None,
                            meta_file=df_dir,
                            prefix="transformer",
                            output_directory=output_dir,
                            shuffle=True,
                            tag="training")
    
    df_meta = pd.read_csv(df_dir, sep="\t")
    bp_resolution = BP_RESOLUTION

    # Get the relevant cell lines in roi pool
    indices = train_examples.ROI_pool[
        (train_examples.ROI_pool["Cell_Line"] == cell_type) & (train_examples.ROI_pool["ROI_Type"] == "ATAC")
    ].index[:]

    input_batch, target_batch = [], []
    moods_batch = []
    for i in indices:

        # Take the relevant cell lines in roi pool
        roi_row = train_examples.ROI_pool.iloc[i, :]

        # Get the paths for the cell type of interest.
        meta_row = df_meta[(df_meta['Cell_Line'] == cell_type)]
        meta_row = meta_row.reset_index(drop=True)

        # Rename some variables. This just helps clean up code downstream
        chrom_name = roi_row['Chr']
        start = int(roi_row['Start'])
        end = int(roi_row['Stop'])

        signal = meta_row.loc[0, 'ATAC_Signal_File']
        binding = meta_row.loc[0, 'Binding_File']

        with \
            load_2bit(sequence) as sequence_stream, \
            load_bigwig(signal) as signal_stream, \
            load_bigwig(binding) as binding_stream:

            # Get the input matrix of values and one-hot encoded sequence
            input_matrix = get_input_matrix(signal_stream=signal_stream,
                                            sequence_stream=sequence_stream,
                                            chromosome=chrom_name,
                                            start=start,
                                            end=end,
                                            )       # Each input_matrix has shape (1024, 5)
            
            if moods_bigwig is not None:
                with load_bigwig(moods_bigwig) as moods:
                    moods_np = np.array(moods.values(chromosome, start, end))   # shape = (1024,)
                    moods_batch.append(moods_np)
            
            input_batch.append(input_matrix)

            try:
                # Get the target matrix
                target_vector = np.array(binding_stream.values(chrom_name, start, end)).T

            except:
                # TODO change length of array
                target_vector = np.zeros(1024)

            # change nan to numbers
            target_vector = np.nan_to_num(target_vector, 0.0)

            # get the number of 32 bp bins across the input sequence
            n_bins = int(target_vector.shape[0] / bp_resolution)

            # Split the data up into 32 x 32 bp bins.
            split_targets = np.array(np.split(target_vector, n_bins, axis=0))

            # TODO we might want to test what happens if we change the
            bin_sums = np.sum(split_targets, axis=1)
            bin_vector = np.where(bin_sums > 0.5 * bp_resolution, 1.0, 0.0)

            target_batch.append(bin_vector)

    input_batch = np.array(input_batch)
    target_batch = np.array(target_batch)

    if moods_bigwig is not None:
        moods_batch = np.array(moods_batch)
        return (input_batch, target_batch, moods_batch)
    else:
        return (input_batch, target_batch)


def get_ism_data(orig_seq):
    """
    Get the ISM data mutated at each base
    orig_seq has shape (1, 1024, 4)
    """
    if len(orig_seq.shape) == 2:
        orig_seq = np.expand_dims(orig_seq, axis=0)
    seq_len = orig_seq.shape[1]
    ism_data = np.zeros((4*seq_len, seq_len, 4))

    # Help me think of doing this without using loops
    for i in range(seq_len):
        identity = np.eye(4, 4)
        mutated = np.repeat(orig_seq, repeats=4, axis=0)
        mutated[:, i, :] = identity
        ism_data[i*4:(i+1)*4, :, :] = mutated

    return ism_data


def get_intermediate_layer_names(model_base_dir, model_config, branch):
    """
    Based on the name of the model, return the associated layer names
    Two model types are currently supported:
        - multiinput_<tf>_512_new
        - multiinput_<tf>_512_enformerconv
    """
    model_name = ntpath.basename(model_base_dir)
    model = load_model_from_dir(model_base_dir, model_config)

    if "multiinput_" in model_name and "_512_new" in model_name:
        if branch == "atac":
            all_layer_names = []
            all_layer_names.append([layer.name for layer in model.layers if "atac" in layer.name][-1])
            all_layer_names.append("Intermediate_fusion_conv_relu_1")
            all_layer_names.extend(
                [layer.name for layer in model.layers if "Transformer_block" in layer.name]
            )
        elif branch == "genome":
            all_layer_names = []
            all_layer_names.append([layer.name for layer in model.layers if "genome" in layer.name][-1])
            all_layer_names.append("Intermediate_fusion_conv_relu_1")
            all_layer_names.extend(
                [layer.name for layer in model.layers if "Transformer_block" in layer.name]
            )

    elif "multiinput_" in model_name and "_512_enformerconv" in model_name:
        if branch == "atac":
            all_layer_names = ["atac_maxpool_1", "atac_maxpool_2"]
            all_layer_names.extend(
                [layer.name for layer in model.layers if "Transformer_block" in layer.name]
            )
        elif branch == "genome":
            all_layer_names = ["genome_maxpool_1", "genome_maxpool_2"]
            all_layer_names.extend(
                [layer.name for layer in model.layers if "Transformer_block" in layer.name]
            )

    return all_layer_names


def visualize_latent_vectors(latent_vectors, all_layer_names, output_prefix):
    """
    Create heatmap to visualize the latent vectors
    """
    fig, axes = plt.subplots(nrows=len(all_layer_names), ncols=1, figsize=(15, 15))
    num_samples = latent_vectors.shape[0]
    for j in range(num_samples):
        fig, axes = plt.subplots(nrows=len(all_layer_names), ncols=1, figsize=(15, 15))
        for i in range(len(all_layer_names)):
            layer_name = all_layer_names[i]
            latent_vector = latent_vectors[j, i, ...]
            sns.heatmap(latent_vector.T, ax=axes[i])
            axes[i].set_title(layer_name)
        plt.tight_layout()
        plt.savefig(
            f"{output_prefix}-id{j}.png"
        )


def run_ism(orig_seq, model, bases_of_interest=[], batch_size=128, output_size=32, seq_len=1024, inter_fusion=True):
    """
    Run ISM and return the vector of difference
    """
    #logging.error(orig_seq.shape)
    one_hot_encoded_genome = orig_seq[:, :, :4]
    atac_signal = np.expand_dims(orig_seq[:, :, -1], axis=-1)

    # Create a vector of shape (1024x4, 1024, 5)
    mutagenesis_vector = get_ism_data(one_hot_encoded_genome)
    mutagenesis_vector = np.concatenate([mutagenesis_vector, np.repeat(atac_signal, repeats=mutagenesis_vector.shape[0], axis=0)], axis=-1)

    # Forward pass the original sequence
    orig_input = input_for_interfusion_ism(orig_seq, inter_fusion)
    #logging.error(type(orig_input))
    #logging.error(orig_input["genome"].shape)
    orig_output = model.predict(orig_input)   # orig_output.shape = (1, 32)

    # Work through the outputs in batch of 128
    num_batches = mutagenesis_vector.shape[0] // batch_size
    ism_outputs = np.zeros((mutagenesis_vector.shape[0], output_size))
    for i in range(num_batches):
        ism_input = mutagenesis_vector[i*batch_size:(i+1)*batch_size, :, :]
        ism_input = input_for_interfusion_ism(ism_input, inter_fusion)
        ism_output = model.predict(ism_input)       # ism_output.shape = (128, 32)
        ism_outputs[i*batch_size:(i+1)*batch_size, :] = ism_output

    # Take the difference between the original output and the new outputs at the bases specified in base_of_interest
    differences = []
    for base in bases_of_interest:
        # Let's try taking the absolute value
        difference = np.abs(ism_outputs[:, base] - orig_output[:, base])    # Broadcasting, shape = (1024x4,len(base_of_interest))
        difference = np.squeeze(difference)         # shape = (1024x4,)
        difference = np.reshape(difference, (seq_len, 4))   # shape = (1024, 4)
        differences.append(difference)

    return differences


def run_shap_genome(input_sample, model, bases_of_interest, num_background_seqs=10, interfusion=True):
    """
    input_sample is a list of [genome, signal]
    Run IG providing a set of shuffled genomes as background
    """
    genome, signal = input_sample    # [shape=(1, seq_len, 4), shape=(1, seq_len, 1)]

    # get background data
    background = dinuc_shuffle_DNA_only_several_times(genome, signal, num_background_seqs, interfusion)
    
    # Create new model with only one output
    try:
        new_model = tf.keras.Model(model.input, model.layers[-1].output[:, bases_of_interest])
    except:
        new_model = tf.keras.Model(model.input, model.layers[-1].output[:, bases_of_interest[0]])
    
    # Run IG
    explainer = GradientExplainer(
        model=new_model,
        data=background
    )
    if interfusion:
        contribs = explainer.shap_values([genome, signal])
    else:
        contribs = explainer.shap_values(np.concatenate([genome, signal], axis=-1))
    return contribs


def run_shap_atac(input_sample, model, bases_of_interest, num_background_seqs=10, interfusion=True):
    """
    input_sample is a list of [genome, signal]
    Run IG providing an intepolated signal from 0 to current signal as background
    """
    genome, signal = input_sample    # [shape=(1, seq_len, 4), shape=(1, seq_len, 1)]

    # get background data
    background = background_atac_for_IG(genome, signal, num_background_seqs, interfusion)
    
    # Create new model with only one output
    try:
        new_model = tf.keras.Model(model.input, model.layers[-1].output[:, bases_of_interest])
    except:
        new_model = tf.keras.Model(model.input, model.layers[-1].output[:, bases_of_interest[0]])
    
    # Run IG
    explainer = GradientExplainer(
        model=new_model,
        data=background
    )
    if interfusion:
        contribs = explainer.shap_values([genome, signal])
    else:
        contribs = explainer.shap_values(np.concatenate([genome, signal], axis=-1))
    return contribs


def input_maximization_atac_interfusion(true_inputs, true_outputs, model_dir):
    """
    Maximize the ATAC signal for multiinput models
    Returns: a numpy array of ATAC signals that maximize the outputs
    """
    # Some constants
    epochs = 50

    # Prepare the inputs
    true_inputs = input_for_interfusion_ism(true_inputs, True)

    # Prepare the trainable inputs
    signals = true_inputs["atac"]
    trainable_signals = tf.Variable(signals, trainable=True)
    trainable_inputs = {"genome": true_inputs["genome"], "atac": trainable_signals}

    # load and freeze the model
    model = load_model_from_dir(model_dir)
    model.trainable = False

    # define the loss function and optimizer
    loss_func = loss_function
    optimizer = tf.keras.optimizers.Adam()

    # take the gradient wrt to the input
    for epoch in epochs:
        with tf.GradientTape() as tape:
            tape.watch(trainable_signals)
            preds = model(trainable_inputs, training=False)
            loss = loss_func(true_outputs, preds)
        grad = tape.gradient(loss, trainable_signals)
        optimizer.apply_gradient(zip(grad, trainable_signals))

    return trainable_signals.numpy()

