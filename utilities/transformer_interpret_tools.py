import tensorflow as tf
import numpy as np
import json
import os
import pandas as pd

from maxatac.utilities.training_tools import DataGenerator, get_input_matrix
from maxatac.utilities.genome_tools import load_bigwig, load_2bit
from maxatac.utilities.constants import BP_RESOLUTION

from maxatac.utilities.training_tools import MaxATACModel, ROIPool

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

def load_model_from_dir(model_base_dir):
    """
    Load the model from dir
    """
    with open(os.path.join(model_base_dir, "best_epoch.txt"), "r") as f:
        best_model_dir = f.read().strip()

    with open(os.path.join(model_base_dir, "cmd_args.json"), "rb") as f:
        train_args = json.load(f)

    model = MaxATACModel(
        arch=train_args["arch"],
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


def get_data(meta_file, chromosome, cell_type, output_dir):
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
                                            )
            
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
    return (input_batch, target_batch)