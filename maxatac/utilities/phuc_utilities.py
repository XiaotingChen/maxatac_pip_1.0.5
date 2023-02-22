# my file that contains all random functions and ideas
from contextlib import redirect_stdout
import logging
import os
import pandas as pd
import sys
import numpy as np
from scipy.spatial import distance
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from maxatac.utilities.system_tools import get_dir, Mute

with Mute():
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from tensorflow.keras.utils import plot_model
    from maxatac.utilities.genome_tools import build_chrom_sizes_dict
    from maxatac.utilities.prediction_tools import write_predictions_to_bigwig, \
        import_prediction_regions, create_prediction_regions, make_stranded_predictions, PredictionDataGenerator
    from maxatac.utilities.constants import DATA_PATH, INPUT_CHANNELS
    from maxatac.analyses.peaks import run_call_peaks
    from maxatac.architectures.dcnn import get_dilated_cnn, get_dilated_cnn_with_attention, dice_coef
    from maxatac.architectures.transformers import get_transformer
    from maxatac.utilities.constants import TRAIN_SCALE_SIGNAL, BLACKLISTED_REGIONS, DEFAULT_CHROM_SIZES, INPUT_LENGTH

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
            if tf_true and cell_true: return os.path.join(data_dir, folder_name, folder_name, filename)

        return None

    # Create list of cell lines and tfs
    cell_lines, tfs = [], []
    with open(os.path.join(data_dir, "cell_lines.txt"), "r") as f:
        for line in f.readlines():
            cell_lines.append(line.rstrip("\n"))
    with open(os.path.join(data_dir, "tf.txt"), "r") as f:
        for line in f.readlines():
            tfs.append(line.rstrip("\n"))

    files = []
    for cell_line in cell_lines:
        for tf in tfs:
            row = [
                cell_line,
                tf,
                find_files("ChIP_Binding_File", cell_line=cell_line, tf=tf),
                find_files("ChIP_Peaks", cell_line=cell_line, tf=tf),
                find_files("ATAC_Signal_File", cell_line=cell_line, tf=""),
                find_files("ATAC_Peaks", cell_line=cell_line, tf=""),
                "Train"     # let's put all row as train for now
            ]
            if None not in row:
                files.append(row)

    df = pd.DataFrame(data=files, columns=["Cell_Line", "TF", "CHIP_Peaks", "Binding_File", "ATAC_Signal_File", "ATAC_Peaks", "Train_Test_Label"])
    df.to_csv(os.path.join(data_dir, "meta_file.tsv"), sep="\t")
    print(f"Metafile located at {os.path.join(data_dir, 'meta_file.tsv')}")


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
        with open("/data/weirauchlab/team/ngun7t/maxatac/runs/model_summary.txt", "w") as f:
            with redirect_stdout(f):
                nn_model.summary()
    except:
        logging.error("The model does not exist")


def debug_plot_model(model_link):
    """
    Get a random model from model_link and and test the plot_model function
    """
    model = load_model(model_link, compile=False)
    plot_model(model, to_file="/data/weirauchlab/team/ngun7t/maxatac/temp/test_plot_model.png")
    print(f"The model is saved at /data/weirauchlab/team/ngun7t/maxatac/temp/test_plot_model.png")


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
    toy_inputs = np.zeros((10, 1024, 5))    # batch_size x seq_len x (one hot encoded DNA + ATACseq signal)

    if model_name == "DCNN_V2":
        model = get_dilated_cnn(output_activation=output_activation,
                                target_scale_factor=target_scale_factor,
                                dense_b=dense,
                                weights=weights
                                )
 
    elif model_name == "DCNN_V2_attention":
        model = get_dilated_cnn_with_attention(
            output_activation=output_activation,
            target_scale_factor=target_scale_factor,
            dense_b=dense,
            weights=weights
        )

    elif model_name == "Transformer_phuc":
        model = get_transformer(
            output_activation=output_activation,
            target_scale_factor=target_scale_factor,
            dense_b=dense,
            weights=weights
        )
    else:
        sys.exit("Model Architecture not specified correctly. Please check")

    outputs = model.predict(toy_inputs)
    print(f"Output shape: {outputs.shape}")


def ablation_random_genome(metadata_file):
    """
    Keep ATACseq constant, randomize genome sequence, check if the prediction is driven by the genome or the ATACseq
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
            if genome[i] == "A": one_hot_encode_genome[i, 0] = 1
            if genome[i] == "T": one_hot_encode_genome[i, 1] = 1
            if genome[i] == "G": one_hot_encode_genome[i, 2] = 1
            if genome[i] == "C": one_hot_encode_genome[i, 3] = 1
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
    #if args.chromosomes[0] == 'all':
    #    from maxatac.utilities.constants import AUTOSOMAL_CHRS as all_chr
    #    args.chromosomes = all_chr


    # Output filename for the bigwig predictions file based on the output directory and the prefix. Add the bw extension
    #outfile_name_bigwig = os.path.join(output_directory, args.prefix + ".bw")

    # The function build_chrom_sizes_dict is used to make sure regions fall within chromosome bounds.
    # Create a dictionary of chromosome sizes used to make the bigwig files
    chrom_sizes_dict = build_chrom_sizes_dict(chromosomes, DEFAULT_CHROM_SIZES)
    
    # Import the regions for prediction.
    regions_pool = create_prediction_regions(chromosomes=chromosomes,
                                                chrom_sizes=DEFAULT_CHROM_SIZES,
                                                blacklist=BLACKLISTED_REGIONS,
                                                step_size=int(INPUT_LENGTH/4)
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
    data_generator = PredictionDataGenerator(signal=signal,
                                             sequence=sequence,
                                             input_channels=INPUT_CHANNELS,
                                             input_length=INPUT_LENGTH,
                                             predict_roi_df=chr_roi_pool,
                                             batch_size=batch_size,
                                             use_complement=False)
    
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
    plt.savefig(os.path.join(folder_full_dir, f"{cell_line}_{tf}_{chromosome}_histplot.png"))

    while np.amax(total_atac) <= 0.0:
        inputs = next(iter(data_generator))
        total_atac = inputs[:, :, -1]
        print(np.amax(total_atac))

    # Get the number of bins from a batch 
    largest_atac_peak = np.amax(max_atac_signals)
    slices = []
    for i in reversed(range(int(largest_atac_peak)+1)):
        # sample the indices at which the max input ATAC signal falls within the bin
        inds = np.where(np.logical_and(
            max_atac_signals > i, max_atac_signals < i+1
        ))[0]
        if (inds.size == 0): continue
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
        orig_output = model.predict(inp, verbose=0) # shape (1, 32)
        #print(f"orig_output: {orig_output.shape}")

        # Roll the ATAC-seq signal (in batches of 16)
        for i in range(0, seq_len, num_samples_per_random):
            new_atacs = [np.roll(np.squeeze(atac), k + i*num_samples_per_random) for k in range(num_samples_per_random)]
            new_atacs = np.expand_dims(np.stack(new_atacs), axis=-1) # shape is hopefully (batch_size, seq_len, 1)
            new_inputs = np.concatenate(
                [np.repeat(orig_genome, num_samples_per_random, axis=0), new_atacs], axis=-1
            )
            new_output = model.predict(new_inputs, verbose=0)

            # Calculate distances
            for b in range(num_samples_per_random):
                vector = new_output[b].T
                dist_atac[j, b+i] = distance.euclidean(np.squeeze(vector), np.squeeze(orig_output))
  

        # Random genome shuffling
        for i in range(num_epoch):

            random_input = generate_random_input(seq_len, num_samples_per_random)
            new_output = model.predict(random_input, verbose=0)        # shape: (batch_size, 32)
            #print(f"new_output shape: {new_output.shape}")

            # Calculate distances
            for b in range(num_samples_per_random):
                vector = new_output[b].T
                dist_genome[j, b+i*num_samples_per_random] = distance.euclidean(np.squeeze(vector), np.squeeze(orig_output))
      

    # Plots
    # There are two plots: the plot of the original atac-seq signal and the cosine plot
    columns = list(max_atac_signals[slices])
    columns = [str(f"{i:.2f}") for i in columns]
    df_atac = pd.DataFrame(
        data=dist_atac.T, columns=columns
    )
    df_atac.to_csv(os.path.join(folder_full_dir, "atac.csv"), index=False)
    new_df_atac = pd.read_csv(os.path.join(folder_full_dir, "atac.csv"))
    plt.figure()
    sns.boxplot(data=new_df_atac)
    plt.xlabel("Bases shifted")
    plt.ylabel("Euclidean distance")
    plt.xticks(rotation=90)
    plt.title("Change of output with ATAC-seq shifting")
    plt.savefig(os.path.join(folder_full_dir, f"{cell_line}_{tf}_{chromosome}_boxplot_atac.png"))

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
    plt.savefig(os.path.join(folder_full_dir, f"{cell_line}_{tf}_{chromosome}_boxplot_genome.png"))
    print("Done")
