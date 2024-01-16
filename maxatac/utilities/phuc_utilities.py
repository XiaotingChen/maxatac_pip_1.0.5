# my file that contains all random functions and ideas
from contextlib import redirect_stdout
import logging
import os
import json
import pandas as pd
import sys
import numpy as np
from scipy.spatial import distance
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import ntpath
from sklearn.metrics import precision_recall_curve, confusion_matrix

from maxatac.utilities.system_tools import Mute

with Mute():
    from tensorflow.keras.models import load_model
    from maxatac.utilities.training_tools import DataGenerator, ROIPool
    from maxatac.utilities.genome_tools import load_bigwig

with Mute():
    import tensorflow as tf
    from tensorflow.keras.utils import plot_model
    from maxatac.utilities.genome_tools import (
        build_chrom_sizes_dict,
        chromosome_blacklist_mask,
    )
    from maxatac.utilities.prediction_tools import (
        create_prediction_regions,
        PredictionDataGenerator,
    )
    from maxatac.architectures.dcnn import (
        get_dilated_cnn,
        get_dilated_cnn_with_attention,
    )
    from maxatac.architectures.transformers import get_transformer
    from maxatac.utilities.constants import (
        TRAIN_SCALE_SIGNAL,
        BLACKLISTED_REGIONS,
        DEFAULT_CHROM_SIZES,
        INPUT_LENGTH,
        DATA_PATH,
        INPUT_CHANNELS,
        DEFAULT_BENCHMARKING_BIN_SIZE,
        DEFAULT_BENCHMARKING_AGGREGATION_FUNCTION,
        BLACKLISTED_REGIONS_BIGWIG,
    )


def phuc_func(args):
    if args.summary != "":
        get_model_summary(args.summary)
    if args.metafile_data_dir != "":
        prepare_metafile(args.metafile_data_dir)
    if args.debug_plot_model != "":
        debug_plot_model(args.debug_plot_model)
    if args.debug_forward_pass_model != "":
        debug_model_architectures(args.debug_forward_pass_model)
    if args.ablation_random_genome_file != "":
        ablation_random_genome(args.ablation_random_genome_file)
    if args.compare_training_and_zorn != []:
        visualize_zorn_atac(args.compare_training_and_zorn)
    if args.count_peaks != []:
        peak_analysis(args.count_peaks)
    if args.atac_jointplot != []:
        atac_signal_jointplot(args.atac_jointplot)


def prepare_metafile(data_dir):
    """
    Prepare the metafile necessary to feed into the model for training
    Arguments:
        - data_dir: The path to the data folder. The name of the folders is the same as the name of the columns to which the files belong
    """

    def find_files(folder_name, cell_line="", tf=""):
        for filename in os.listdir(os.path.join(data_dir, folder_name, folder_name)):
            tf_true = (tf in filename) or (tf == "")
            cell_true = (cell_line in filename) or (cell_line == "")
            if tf_true and cell_true:
                return os.path.join(data_dir, folder_name, folder_name, filename)

        return None

    # Create list of cell lines and tfs
    tfs = []
    with open(os.path.join(data_dir, "tf.txt"), "r") as f:
        for line in f.readlines():
            tfs.append(line.rstrip("\n"))

    with open(os.path.join(data_dir, "tf_to_cell_line.json"), "r") as f:
        d = json.load(f)

    for tf in tfs:
        files = []
        cell_lines = d[tf]
        for cell_line in cell_lines:
            row = [
                cell_line,
                tf,
                find_files("ChIP_Binding_File", cell_line=cell_line, tf=tf),
                find_files("ChIP_Peaks", cell_line=cell_line, tf=tf),
                find_files("ATAC_Signal_File", cell_line=cell_line, tf=""),
                find_files("ATAC_Peaks", cell_line=cell_line, tf=""),
                "Train",  # let's put all row as train for now
            ]
            if None not in row:
                files.append(row)

        df = pd.DataFrame(
            data=files,
            columns=[
                "Cell_Line",
                "TF",
                "CHIP_Peaks",
                "Binding_File",
                "ATAC_Signal_File",
                "ATAC_Peaks",
                "Train_Test_Label",
            ],
        )
        df.to_csv(os.path.join(data_dir, f"meta_file_{tf}.tsv"), sep="\t")
        print(
            f"Metafile for {tf} located at {os.path.join(data_dir, f'meta_file_{tf}.tsv')}"
        )


def generate_numpy_arrays(gen, save_dir):
    """
    (beta) Generate numpy arrays for convenient analysis
    This function takes the data generator from analyses/train.py and yield every item
    There can be memory issues
    Params:
        - gen: a generator, each yield returns a batch of X and y
    """

    def extract(gen):
        yield from gen

    for i, (X, y) in enumerate(extract(gen)):
        pass


def get_model_summary(model_link):
    """
    Get the model from model_link and print it to the .txt files
    """
    try:
        nn_model = load_model(model_link, compile=False)
        with open(
            "/data/weirauchlab/team/ngun7t/maxatac/runs/model_summary.txt", "w"
        ) as f:
            with redirect_stdout(f):
                nn_model.summary()
    except:
        try:
            pass
        except:
            logging.error("The model does not exist")


def debug_plot_model(model_link):
    """
    Get a random model from model_link and and test the plot_model function
    """
    model = load_model(model_link, compile=False)
    plot_model(
        model, to_file="/data/weirauchlab/team/ngun7t/maxatac/temp/test_plot_model.png"
    )
    print(
        f"The model is saved at /data/weirauchlab/team/ngun7t/maxatac/temp/test_plot_model.png"
    )


def debug_model_architectures(model_name):
    """
    Try out different model architectures and see if they work for the forward pass
    Read from the input.txt file some important info
    """

    # Based on the default values in the parser
    output_activation = "sigmoid"
    target_scale_factor = TRAIN_SCALE_SIGNAL
    dense = False
    weights = None
    toy_inputs = np.zeros(
        (10, 1024, 5)
    )  # batch_size x seq_len x (one hot encoded DNA + ATACseq signal)

    if model_name == "DCNN_V2":
        model = get_dilated_cnn(
            output_activation=output_activation,
            target_scale_factor=target_scale_factor,
            dense_b=dense,
            weights=weights,
        )

    elif model_name == "DCNN_V2_attention":
        model = get_dilated_cnn_with_attention(
            output_activation=output_activation,
            target_scale_factor=target_scale_factor,
            dense_b=dense,
            weights=weights,
        )

    elif model_name == "Transformer_phuc":
        model = get_transformer(
            output_activation=output_activation,
            target_scale_factor=target_scale_factor,
            dense_b=dense,
            weights=weights,
        )
    else:
        sys.exit("Model Architecture not specified correctly. Please check")

    outputs = model.predict(toy_inputs)
    print(f"Output shape: {outputs.shape}")


def _get_generator(signal, chromosome):
    """
    Get a generator for the training data
    """
    sequence = "/users/ngun7t/opt/maxatac/data/hg38/hg38.2bit"
    batch_size = 10000
    chromosomes = ["chr1", "chr8"]

    # predict on all chromosomes
    # if args.chromosomes[0] == 'all':
    #    from maxatac.utilities.constants import AUTOSOMAL_CHRS as all_chr
    #    args.chromosomes = all_chr

    # Output filename for the bigwig predictions file based on the output directory and the prefix. Add the bw extension
    # outfile_name_bigwig = os.path.join(output_directory, args.prefix + ".bw")

    # The function build_chrom_sizes_dict is used to make sure regions fall within chromosome bounds.
    # Create a dictionary of chromosome sizes used to make the bigwig files
    # chrom_sizes_dict = build_chrom_sizes_dict(chromosomes, DEFAULT_CHROM_SIZES)

    # Import the regions for prediction.
    regions_pool = create_prediction_regions(
        chromosomes=chromosomes,
        chrom_sizes=DEFAULT_CHROM_SIZES,
        blacklist=BLACKLISTED_REGIONS,
        step_size=int(INPUT_LENGTH / 4),
    )  # this is a pandas Dataframe

    # chrom_list = chromosomes

    # Get the roi pools on only the chromosomes specified
    chr_roi_pool = regions_pool[regions_pool["chr"] == chromosome].copy()

    # This returns a numpy array
    data_generator = PredictionDataGenerator(
        signal=signal,
        sequence=sequence,
        input_channels=INPUT_CHANNELS,
        input_length=INPUT_LENGTH,
        predict_roi_df=chr_roi_pool,
        batch_size=batch_size,
        use_complement=False,
    )
    return data_generator


def _get_atac_data(chromosome, cell_type, dataset):
    """
    Get atac data
    """

    def meta_file_selection(dataset, cell_type):
        if dataset == "train":
            df_lef1 = pd.read_csv(
                os.path.join(base_dir, "meta_file_LEF1.tsv"), sep="\t"
            )
            df_tcf7 = pd.read_csv(
                os.path.join(base_dir, "meta_file_TCF7.tsv"), sep="\t"
            )
            df_tcf7l2 = pd.read_csv(
                os.path.join(base_dir, "meta_file_TCF7L2.tsv"), sep="\t"
            )
            if cell_type in df_lef1["Cell_Line"].unique().tolist():
                return os.path.join(base_dir, "meta_file_LEF1.tsv")
            elif cell_type in df_tcf7["Cell_Line"].unique().tolist():
                return os.path.join(base_dir, "meta_file_TCF7.tsv")
            else:
                return os.path.join(base_dir, "meta_file_TCF7L2.tsv")

        else:
            return "/data/weirauchlab/team/ngun7t/maxatac/meta_file_for_interpreting_LEF1.tsv"

    batch_size = 10000
    base_dir = "/data/weirauchlab/team/ngun7t/maxatac/training_data"

    df_dir = meta_file_selection(dataset, cell_type)
    train_examples = ROIPool(
        chroms=[chromosome],
        roi_file_path=None,
        meta_file=df_dir,
        prefix="transformer",
        output_directory="/data/weirauchlab/team/ngun7t/maxatac/runs/data_viz",
        shuffle=True,
        tag="training",
    )

    df = pd.read_csv(df_dir, sep="\t")
    train_gen = DataGenerator(
        sequence="/users/ngun7t/opt/maxatac/data/hg38/hg38.2bit",
        meta_table=df,
        roi_pool=train_examples.ROI_pool,
        cell_type_list=[cell_type],
        rand_ratio=0,
        chroms=[chromosome],
        batch_size=batch_size,
        shuffle_cell_type=False,
        rev_comp_train=False,
    )

    return train_gen


def visualize_zorn_atac(args):
    """
    Visualize Zorn's normalized ATAC-seq data to see if it's out of distribution
    Legacy: this function will no longer be needed
    """
    # Let's see if we can plot all ATAC-seq signals in one run
    # We have one ATAC-seq for each cell type
    # Plot the histogram of max ATAC-seq peaks for the training data and the Zorn dataset
    mode = args[0]
    zorn_train_cell_types = [args[1]]
    zorn_cell_types = [args[2]]

    training_atac_dir = "/data/weirauchlab/team/ngun7t/maxatac/training_data/ATAC_Signal_File/ATAC_Signal_File"
    zorn_atac_dir = "/data/weirauchlab/team/ngun7t/maxatac/zorn/Zorn_hESC_ATAC/outputs"
    training_atac = [
        os.path.join(training_atac_dir, n)
        for n in os.listdir(training_atac_dir)
        if n.split("_")[0] in zorn_train_cell_types
    ]
    zorn_atac = glob.glob(f"{zorn_atac_dir}/*/maxatac/normalize_bigwig/*.bw")
    save_dir = "/data/weirauchlab/team/ngun7t/maxatac/runs/data_viz"
    chromosome = "chr1"
    threshold = 4
    num_samples = 5000
    os.makedirs(
        f"{save_dir}/max_atac_peaks_2dhist_{threshold}_{mode}_{chromosome}",
        exist_ok=True,
    )

    # Just for debugging
    zorn_atac = [
        i for i in zorn_atac if ntpath.basename(i).split(".")[0] in zorn_cell_types
    ]

    # Borrow the code from ablation_random_genome for creating the generator
    # fig, axes = plt.subplots(ncols=4, nrows=4, figsize=(30, 20))
    # axes = axes.ravel()

    # Create all pairs of (training-zorn)
    pairs = [(i, j) for i in range(len(training_atac)) for j in range(len(zorn_atac))]

    # Create a list of data inputs
    train_max_peaks = []
    zorn_max_peaks = []

    print("Working on training set")
    for i, tr_atac in enumerate(training_atac):
        if threshold == -1:
            print(f">>> Working on {zorn_train_cell_types[i]}")
            data_generator = _get_generator(tr_atac, chromosome)
            inputs = next(iter(data_generator))
            total_atac = inputs[:, :, -1]
            max_atac_signals = np.max(total_atac, axis=1)
            train_max_peaks.append(max_atac_signals)
        else:
            print(f">>> Working on {zorn_train_cell_types[i]}")
            data_generator = _get_atac_data(
                chromosome, zorn_train_cell_types[i], "train"
            )
            samples_collected = 0
            data = np.zeros((num_samples,)) - 1

            while samples_collected <= num_samples:
                input_data, target_data = next(iter(data_generator))

                # only get the indices with some greater value
                target_sum = np.sum(target_data, axis=1)
                if mode == "peak":
                    inds = np.where(target_sum > threshold)[0]
                elif mode == "non_peak":
                    inds = np.where(target_sum == 0)[0]
                max_atac_peaks = np.max(input_data[:, :, -1], axis=1)[inds]

                # add the value to data
                if samples_collected + max_atac_peaks.size <= num_samples:
                    data[
                        samples_collected : samples_collected + max_atac_peaks.size
                    ] = max_atac_peaks
                else:
                    diff = samples_collected + max_atac_peaks.size - num_samples
                    print(
                        f"samples_collected: {samples_collected}, diff: {diff}, size: {max_atac_peaks.size}"
                    )
                    data[samples_collected:] = max_atac_peaks[
                        : max_atac_peaks.size - diff
                    ]

                samples_collected += max_atac_peaks.size
                print(f">>> Current collected samples: {samples_collected}")

            train_max_peaks.append(data)

    print("Working on zorn dataset")
    for i, zo_atac in enumerate(zorn_atac):
        if threshold == -1:
            print(f">>> Working on {zorn_cell_types[i]}")
            data_generator = _get_generator(zo_atac, chromosome)
            inputs = next(iter(data_generator))
            total_atac = inputs[:, :, -1]
            max_atac_signals = np.max(total_atac, axis=1)
            zorn_max_peaks.append(max_atac_signals)
        else:
            print(f">>> Working on {zorn_cell_types[i]}")
            data_generator = _get_atac_data(chromosome, zorn_cell_types[i], "zorn")
            samples_collected = 0
            data = np.zeros((num_samples,)) - 1

            while samples_collected <= num_samples:
                input_data, target_data = next(iter(data_generator))

                # only get the indices with some greater value
                target_sum = np.sum(target_data, axis=1)
                if mode == "peak":
                    inds = np.where(target_sum > threshold)[0]
                elif mode == "non_peak":
                    inds = np.where(target_sum == 0)[0]
                max_atac_peaks = np.max(input_data[:, :, -1], axis=1)[inds]

                # add the value to data
                if samples_collected + max_atac_peaks.size <= num_samples:
                    data[
                        samples_collected : samples_collected + max_atac_peaks.size
                    ] = max_atac_peaks
                else:
                    diff = samples_collected + max_atac_peaks.size - num_samples
                    print(
                        f"samples_collected: {samples_collected}, diff: {diff}, size: {max_atac_peaks.size}"
                    )
                    data[samples_collected:] = max_atac_peaks[
                        : max_atac_peaks.size - diff
                    ]
                samples_collected += max_atac_peaks.size
                print(f">>> Current collected samples: {samples_collected}")

            zorn_max_peaks.append(data)

    # Loop through the pairs and create the relevant plots

    for tr, zo in pairs:
        plt.figure()
        sns.jointplot(x=train_max_peaks[tr], y=zorn_max_peaks[zo])
        tr_cell_type = ntpath.basename(training_atac[tr]).split("_")[0]
        zo_cell_type = ntpath.basename(zorn_atac[zo]).split(".")[0]
        plt.xlabel(tr_cell_type)
        plt.ylabel(zo_cell_type)
        plt.savefig(
            f"{save_dir}/max_atac_peaks_2dhist_{threshold}_{mode}_{chromosome}/{tr_cell_type}.{zo_cell_type}.{threshold}.png"
        )
        plt.close()

    # for training
    # for i, tr_atac in enumerate(training_atac):
    #    file_name = ntpath.basename(tr_atac)
    #    cell_type = file_name.split("_")[0]
    #    data_generator = _get_generator(tr_atac, chromosome)
    #    inputs = next(iter(data_generator))
    #    total_atac = inputs[:, :, -1]


#
#    # Plot the ATAC-seq max signal histogram
#    max_atac_signals = np.max(total_atac, axis=1)
#    sns.histplot(data=max_atac_signals, ax=axes[i])
#    axes[i].set_title(f"training_{cell_type}")
#
## for zorn
# for i, zo_atac in enumerate(zorn_atac):
#    file_name = ntpath.basename(zo_atac)
#    cell_type = file_name.split("_")[0]
#    data_generator = _get_generator(zo_atac, chromosome)
#    inputs = next(iter(data_generator))
#    total_atac = inputs[:, :, -1]
#
#    # Plot the ATAC-seq max signal histogram
#    max_atac_signals = np.max(total_atac, axis=1)
#    sns.histplot(data=max_atac_signals, ax=axes[i + 8])
#    axes[i + 8].set_title(f"zorn_{cell_type}")
#
# axes[7].remove()
# axes[15].remove()


def ablation_random_genome(metadata_file):
    """
    Keep ATACseq constant, randomize genome sequence, check if the prediction is driven by the genome or the ATACseq
    Legacy: this function will no longer be needed
    """

    # Shuffle the genome
    def one_hot_encode(genome):
        """
        One hot encode the genome
        """
        # Let's do this very manually
        genome_length = genome.size
        one_hot_encode_genome = np.zeros((genome_length, 4))
        for i in range(genome_length):
            if genome[i] == "A":
                one_hot_encode_genome[i, 0] = 1
            if genome[i] == "T":
                one_hot_encode_genome[i, 1] = 1
            if genome[i] == "G":
                one_hot_encode_genome[i, 2] = 1
            if genome[i] == "C":
                one_hot_encode_genome[i, 3] = 1
        return one_hot_encode_genome

    def generate_random_input(length, num_samples):
        """
        Generate random input (random genome + fixed atac)
        """
        nucleotides = ["A", "C", "G", "T"]
        final_output = np.zeros((num_samples, length, 5))
        for i in range(num_samples):
            genome = np.random.choice(nucleotides, size=(length,))
            onehot_genome = np.expand_dims(one_hot_encode(genome), axis=0)
            final_output[0] = np.concatenate([onehot_genome, atac], axis=-1)

        return final_output

    # For the input, borrow code from predict.py
    with open(metadata_file, "r") as f:
        lines = [l.rstrip("\n") for l in f.readlines()]
        signal = lines[0]
        tf = lines[1]
        cell_line = lines[2]
        save_dir = lines[3]

    sequence = "/users/ngun7t/opt/maxatac/data/hg38/hg38.2bit"
    batch_size = 10000
    num_samples = 16
    num_random = 128
    num_samples_per_random = 16
    chromosome = "chr8"
    chromosomes = ["chr1", "chr8"]

    # create a new directory
    folder_full_dir = os.path.join(save_dir, f"{cell_line}_{tf}_{chromosome}")
    try:
        os.mkdir(folder_full_dir)
    except:
        pass

    # If the user provides the TF name,
    if tf:
        model_link = glob.glob(os.path.join(DATA_PATH, "models", tf, tf + "*.h5"))[0]

        cutoff_file = glob.glob(os.path.join(DATA_PATH, "models", tf, tf + "*.tsv"))[0]

    else:
        pass

    # predict on all chromosomes
    # if args.chromosomes[0] == 'all':
    #    from maxatac.utilities.constants import AUTOSOMAL_CHRS as all_chr
    #    args.chromosomes = all_chr

    # Output filename for the bigwig predictions file based on the output directory and the prefix. Add the bw extension
    # outfile_name_bigwig = os.path.join(output_directory, args.prefix + ".bw")

    # The function build_chrom_sizes_dict is used to make sure regions fall within chromosome bounds.
    # Create a dictionary of chromosome sizes used to make the bigwig files
    chrom_sizes_dict = build_chrom_sizes_dict(chromosomes, DEFAULT_CHROM_SIZES)

    # Import the regions for prediction.
    regions_pool = create_prediction_regions(
        chromosomes=chromosomes,
        chrom_sizes=DEFAULT_CHROM_SIZES,
        blacklist=BLACKLISTED_REGIONS,
        step_size=int(INPUT_LENGTH / 4),
    )  # this is a pandas Dataframe

    chrom_list = chromosomes

    # Get the roi pools on only the chromosomes specified
    chr_roi_pool = regions_pool[regions_pool["chr"] == chromosome].copy()

    model = load_model(model_link, compile=False)

    # Checking the log error file, it shows that this log appears after the log above displays a lot of times
    # So I think what happened is that because this make_stranded_predictions is wrapped around a Pool multiprocessing,
    # All of the trained TF models for all chrs are loaded first before it starts going to the generator
    logging.error("Start Prediction Generator")

    # This returns a numpy array
    data_generator = PredictionDataGenerator(
        signal=signal,
        sequence=sequence,
        input_channels=INPUT_CHANNELS,
        input_length=INPUT_LENGTH,
        predict_roi_df=chr_roi_pool,
        batch_size=batch_size,
        use_complement=False,
    )

    # Get a sample of a file, should be in shape (batch_size, seq_len, 5)
    inputs = next(iter(data_generator))
    total_atac = inputs[:, :, -1]
    seq_len = inputs.shape[1]
    print(np.amax(total_atac))

    # Plot the ATAC-seq max signal histogram
    max_atac_signals = np.max(total_atac, axis=1)
    plt.figure()
    sns.histplot(data=max_atac_signals)
    plt.title(f"Max value ATAC peaks in batch of {cell_line}_{tf}_{chromosome}")
    plt.xlabel("Peak values")
    plt.savefig(
        os.path.join(folder_full_dir, f"{cell_line}_{tf}_{chromosome}_histplot.png")
    )

    while np.amax(total_atac) <= 0.0:
        inputs = next(iter(data_generator))
        total_atac = inputs[:, :, -1]
        print(np.amax(total_atac))

    # Get the number of bins from a batch
    largest_atac_peak = np.amax(max_atac_signals)
    slices = []
    for i in reversed(range(int(largest_atac_peak) + 1)):
        # sample the indices at which the max input ATAC signal falls within the bin
        inds = np.where(np.logical_and(max_atac_signals > i, max_atac_signals < i + 1))[
            0
        ]
        if inds.size == 0:
            continue
        # if i is already 0, choose a number of inds so that len(slices) should be equal to num_samples
        # Otherwise randomly choose one ind
        if (i == 0) and (len(slices) < num_samples):
            num_samples_needed = num_samples - len(slices)
            ind = list(np.random.choice(inds, num_samples_needed))
            slices.extend(ind)
        else:
            ind = np.random.choice(inds)
            slices.append(ind)
        # check 2 conditions: if len(slices) is equal to num_samples, then break
        if len(slices) >= num_samples:
            break

    np.random.seed(0)

    dist_genome = np.zeros((num_samples, num_random))
    dist_atac = np.zeros((num_samples, seq_len))

    # go through each inp and generate the needed data
    for j, ind in enumerate(slices):
        # take a sample from the input
        inp = np.expand_dims(inputs[ind], axis=0)

        # Let's assume we have the data like this (actually we only need 1 sample)
        seq_len = inp.shape[1]

        # Separate ATAC-seq signal or original genome   (1, seq_len, 1)
        atac = np.expand_dims(inp[:, :, -1], axis=-1)
        orig_genome = inp[:, :, :4]

        # number of random sequences generated
        num_epoch = num_random // num_samples_per_random

        # Make a forward pass of the original data
        orig_output = model.predict(inp, verbose=0)  # shape (1, 32)
        # print(f"orig_output: {orig_output.shape}")

        # Roll the ATAC-seq signal (in batches of 16)
        for i in range(0, seq_len, num_samples_per_random):
            new_atacs = [
                np.roll(np.squeeze(atac), k + i * num_samples_per_random)
                for k in range(num_samples_per_random)
            ]
            new_atacs = np.expand_dims(
                np.stack(new_atacs), axis=-1
            )  # shape is hopefully (batch_size, seq_len, 1)
            new_inputs = np.concatenate(
                [np.repeat(orig_genome, num_samples_per_random, axis=0), new_atacs],
                axis=-1,
            )
            new_output = model.predict(new_inputs, verbose=0)

            # Calculate distances
            for b in range(num_samples_per_random):
                vector = new_output[b].T
                dist_atac[j, b + i] = distance.euclidean(
                    np.squeeze(vector), np.squeeze(orig_output)
                )

        # Random genome shuffling
        for i in range(num_epoch):
            random_input = generate_random_input(seq_len, num_samples_per_random)
            new_output = model.predict(
                random_input, verbose=0
            )  # shape: (batch_size, 32)
            # print(f"new_output shape: {new_output.shape}")

            # Calculate distances
            for b in range(num_samples_per_random):
                vector = new_output[b].T
                dist_genome[j, b + i * num_samples_per_random] = distance.euclidean(
                    np.squeeze(vector), np.squeeze(orig_output)
                )

    # Plots
    # There are two plots: the plot of the original atac-seq signal and the cosine plot
    columns = list(max_atac_signals[slices])
    columns = [str(f"{i:.2f}") for i in columns]
    df_atac = pd.DataFrame(data=dist_atac.T, columns=columns)
    df_atac.to_csv(os.path.join(folder_full_dir, "atac.csv"), index=False)
    new_df_atac = pd.read_csv(os.path.join(folder_full_dir, "atac.csv"))
    plt.figure()
    sns.boxplot(data=new_df_atac)
    plt.xlabel("Bases shifted")
    plt.ylabel("Euclidean distance")
    plt.xticks(rotation=90)
    plt.title("Change of output with ATAC-seq shifting")
    plt.savefig(
        os.path.join(folder_full_dir, f"{cell_line}_{tf}_{chromosome}_boxplot_atac.png")
    )

    # Plots
    # Turn the data into a Dataframe to be compatible with sns.boxplot
    # Flip to make shape (num_random, num_sample)
    df = pd.DataFrame(data=dist_genome.T, columns=columns)
    df.to_csv(os.path.join(folder_full_dir, "genome.csv"), index=False)
    new_df = pd.read_csv(os.path.join(folder_full_dir, "genome.csv"))
    plt.figure()
    sns.boxplot(data=new_df)
    plt.xlabel("Shift value")
    plt.ylabel("Euclidean distance")
    plt.xticks(rotation=90)
    plt.title("Change of output with random genome sequence")
    plt.savefig(
        os.path.join(
            folder_full_dir, f"{cell_line}_{tf}_{chromosome}_boxplot_genome.png"
        )
    )
    print("Done")


def save_pred_as_numpy(args):
    """
    This function takes in the directory of 3 bigwig files: the prediction of the model of interest (pred_1),
    the gold standard (gold), and the prediction of the baseline model (maxATAC) (pred_2)
    The function then reads these bigwig files and saves them as numpy arrays using the same procedure as in
    maxatac benchmark
    """
    bw_pred_1, bw_gold, bw_pred_2, save_dir, prefix, chrom = args

    # Load the bigwig files
    bw_gold = load_bigwig(bw_gold)
    chrom_length = bw_gold.chroms(chrom)
    bw_pred_1 = load_bigwig(bw_pred_1)
    bw_pred_2 = load_bigwig(bw_pred_2)

    # Define the constants
    bin_size = DEFAULT_BENCHMARKING_BIN_SIZE
    agg_func = DEFAULT_BENCHMARKING_AGGREGATION_FUNCTION
    bin_count = int(int(chrom_length) / int(bin_size))
    blacklist_bw = BLACKLISTED_REGIONS_BIGWIG
    blacklist_mask = chromosome_blacklist_mask(
        blacklist_bw, chrom, chrom_length, bin_count
    )

    # Load the bigwig files into numpy arrays
    gold_arr = (
        np.nan_to_num(
            np.array(
                bw_gold.stats(
                    chrom, 0, chrom_length, type=agg_func, nBins=bin_count, exact=True
                ),
                dtype=float,
            )
        )
        > 0
    )
    gold_arr = gold_arr[blacklist_mask]
    pred1_arr = np.nan_to_num(
        np.array(
            bw_pred_1.stats(
                chrom, 0, chrom_length, type=agg_func, nBins=bin_count, exact=True
            ),
            dtype=float,
        )
    )
    pred1_arr = pred1_arr[blacklist_mask]
    pred2_arr = np.nan_to_num(
        np.array(
            bw_pred_2.stats(
                chrom, 0, chrom_length, type=agg_func, nBins=bin_count, exact=True
            ),
            dtype=float,
        )
    )
    pred2_arr = pred2_arr[blacklist_mask]

    os.makedirs(os.path.join(save_dir, "numpy"), exist_ok=True)
    with open(os.path.join(save_dir, "numpy", f"{prefix}_gold.npy"), "wb") as f:
        np.save(f, gold_arr)
    with open(os.path.join(save_dir, "numpy", f"{prefix}_pred_1.npy"), "wb") as f:
        np.save(f, pred1_arr)
    with open(os.path.join(save_dir, "numpy", f"{prefix}_pred_1.npy"), "wb") as f:
        np.save(f, pred2_arr)


def peak_analysis(args):
    """
    This function does some of the following:
        - Load the bigwig files with the same methods as the benchmark code
        - Decide peaks based on a certain threshold
        - Generate a confusion matrix file
        - Create a bw file or bed file containing false pos and false neg peaks
    """
    bw_pred, bw_gold, bw_pred_2, save_dir, prefix = args

    def bw_preprocess(
        signal_pred,
        signal_gold,
        signal_pred_2,
        chromosome,
        chrom_length,
        save_numpy=True,
    ):
        bin_size = DEFAULT_BENCHMARKING_BIN_SIZE
        agg_func = DEFAULT_BENCHMARKING_AGGREGATION_FUNCTION
        recall_range = [0.049, 0.051]

        bin_count = int(int(chrom_length) / int(bin_size))
        blacklist_bw = BLACKLISTED_REGIONS_BIGWIG
        blacklist_mask = chromosome_blacklist_mask(
            blacklist_bw, chromosome, chrom_length, bin_count
        )

        gold_arr = (
            np.nan_to_num(
                np.array(
                    signal_gold.stats(
                        chromosome,
                        0,
                        chrom_length,
                        type=agg_func,
                        nBins=bin_count,
                        exact=True,
                    ),
                    dtype=float,
                )
            )
            > 0
        )
        gold_arr = gold_arr[blacklist_mask]

        pred_arr = np.nan_to_num(
            np.array(
                signal_pred.stats(
                    chromosome,
                    0,
                    chrom_length,
                    type=agg_func,
                    nBins=bin_count,
                    exact=True,
                ),
                dtype=float,
            )
        )

        pred_arr = pred_arr[blacklist_mask]

        if save_numpy:
            os.makedirs(os.path.join(save_dir, "numpy"), exist_ok=True)
            with open(os.path.join(save_dir, "numpy", f"{prefix}_gold.npy"), "wb") as f:
                np.save(f, gold_arr)
            with open(
                os.path.join(save_dir, "numpy", f"{prefix}_pred_1.npy"), "wb"
            ) as f:
                np.save(f, pred_arr)

        precision, recall, thresholds = precision_recall_curve(gold_arr, pred_arr)
        df = pd.DataFrame(
            {
                "Precision": precision[:-1],
                "Recall": recall[:-1],
                "Threshold": thresholds,
            }
        )
        df_recall_range = df[
            (df["Recall"] < recall_range[1]) & (df["Recall"] > recall_range[0])
        ]
        threshold = df_recall_range["Threshold"].median()

        # a better way is to keep a list of the thresholds, then run this for all of the thresholds
        pred_arr[pred_arr >= threshold] = 1
        pred_arr[pred_arr < threshold] = 0

        pred_arr_2 = None
        if signal_pred_2 is not None:
            pred_arr_2 = np.nan_to_num(
                np.array(
                    signal_pred_2.stats(
                        chromosome,
                        0,
                        chrom_length,
                        type=agg_func,
                        nBins=bin_count,
                        exact=True,
                    ),
                    dtype=float,
                )
            )
            pred_arr_2 = pred_arr_2[blacklist_mask]
            if save_numpy:
                with open(
                    os.path.join(save_dir, "numpy", f"{prefix}_pred_2.npy"), "wb"
                ) as f:
                    np.save(f, pred_arr_2)

            precision, recall, thresholds = precision_recall_curve(gold_arr, pred_arr_2)
            df = pd.DataFrame(
                {
                    "Precision": precision[:-1],
                    "Recall": recall[:-1],
                    "Threshold": thresholds,
                }
            )
            df_recall_range = df[
                (df["Recall"] < recall_range[1]) & (df["Recall"] > recall_range[0])
            ]
            threshold = df_recall_range["Threshold"].median()

            # a better way is to keep a list of the thresholds, then run this for all of the thresholds
            pred_arr_2[pred_arr_2 >= threshold] = 1
            pred_arr_2[pred_arr_2 < threshold] = 0

        return (pred_arr, gold_arr, pred_arr_2)

    chrom = "chr1"

    bw_gold_file = load_bigwig(bw_gold)
    chrom_length = bw_gold_file.chroms(chrom)
    bw_pred_file = load_bigwig(bw_pred)
    if bw_pred_2 is not None:
        bw_pred_file_2 = load_bigwig(bw_pred_2)

    # Get the preprocessed bw for creating confusion matrix
    pred_1, gold, pred_2 = bw_preprocess(
        bw_pred_file, bw_gold_file, bw_pred_file_2, chrom, chrom_length
    )

    # Get confusion matrix
    tn1, fp1, fn1, tp1 = confusion_matrix(gold, pred_1).ravel()
    data_rows = []
    data_rows.append([bw_pred, bw_gold, None, tp1, tn1, fp1, fn1])

    if pred_2 is not None:
        tn2, fp2, fn2, tp2 = confusion_matrix(gold, pred_2).ravel()
        data_rows.append([None, bw_gold, bw_pred_2, tp2, tn2, fp2, fn2])

        tn3, fp3, fn3, tp3 = confusion_matrix(pred_1, pred_2).ravel()
        data_rows.append([bw_pred, None, bw_pred_2, tp3, tn3, fp3, fn3])

    df = pd.DataFrame(
        data=np.array(data_rows),
        columns=["Pred_1", "Gold", "Pred_2", "TP", "TN", "FP", "FN"],
    )
    os.makedirs(os.path.join(save_dir, "csv"), exist_ok=True)
    df.to_csv(os.path.join(save_dir, "csv", f"{prefix}.csv"), index=False)


def atac_signal_jointplot(output_dir):
    """
    Generate a jointplot that:
        - The x-axis contains the value from 0 to 1024, equal to input vector size
        - The y-axis contains the distribution of ATAC-seq signals
    This method is used to visualize the distribution of ATAC-seq signals on each base
    of the input vector
    For example, we have a sample of N ATAC-seq tracks of (N, 1024)
    We now can plot 1024 distributions of ATAC-seq signals, one for each base,
    each dist containing N samples
    """

    # Currently in debug
    with open("/data/weirauchlab/team/ngun7t/maxatac/scratch/input.npy", "rb") as f:
        inputs = np.load(f)
    with open("/data/weirauchlab/team/ngun7t/maxatac/scratch/target.npy", "rb") as f:
        targets = np.load(f)

    # atac-seq signal
    atac_seq = inputs[:, :, -1]  # shape (N, 1024)

    # create a pandas dataframe to feed to sns jointplot
    num_seq, seq_len = atac_seq.shape
    atac_seq_flat = atac_seq.flatten()
    inds = np.tile(np.arange(seq_len), num_seq)
    df = pd.DataFrame(data={"atac_seq_values": atac_seq_flat, "base": inds})

    plt.figure()
    sns.jointplot(data=df, x="base", y="atac_seq_values", kind="hex")
    plt.savefig("/data/weirauchlab/team/ngun7t/maxatac/scratch/jointplot.png")
