# This is the file for interpretation of transformer models
import tensorflow as tf
import numpy as np
import os
import json
import ntpath
from shap import DeepExplainer, GradientExplainer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import logging
from scipy.stats import zscore

from maxatac.utilities.transformer_interpret_tools import (
    dinuc_shuffle,
    get_data,
    load_model_from_dir,
    get_ism_data,
    input_for_interfusion_att,
    input_for_interfusion_ism,
    run_ism,
    input_maximization_atac_interfusion,
    run_shap_genome,
    run_shap_atac,
    get_intermediate_layer_names,
    visualize_latent_vectors,
)
from maxatac.utilities.plot import plot_ism_results


def run_transformer_interpret(args):
    """
    The main place for all transformer interpretation functions
    """
    if args.analysis == "":
        return

    if args.analysis == "shap":
        attribution_shap_genome(
            args.meta_file,
            args.model_config,
            args.chromosome,
            args.cell_type,
            args.output_dir,
            args.model_base_dir,
        )

    if args.analysis == "latent_viz":
        conv_tower_output(
            args.meta_file,
            args.model_config,
            args.chromosome,
            args.cell_type,
            args.output_dir,
            args.model_base_dir,
            technique=args.dim_reduction_technique,
        )

    if args.analysis == "check_offset":
        check_attention_offset_distribution(
            args.meta_file,
            args.model_config,
            args.chromosome,
            args.cell_type,
            args.output_dir,
            args.model_base_dir,
        )

    if args.analysis == "ISM":
        ISM(
            args.meta_file,
            args.model_config,
            args.chromosome,
            args.cell_type,
            args.output_dir,
            args.model_base_dir,
        )

    if args.analysis == "check_trans_contrib":
        check_layer_contributions(
            args.meta_file,
            args.model_config,
            args.chromosome,
            args.cell_type,
            args.output_dir,
            args.model_base_dir,
        )

    if args.analysis == "attention_weights_viz":
        plot_attention_weights(
            args.meta_file,
            args.model_config,
            args.chromosome,
            args.cell_type,
            args.output_dir,
            args.model_base_dir,
        )

    if args.analysis == "ism_att":
        ism_attention_weights_correlation(
            args.meta_file,
            args.model_config,
            args.chromosome,
            args.cell_type,
            args.output_dir,
            args.model_base_dir,
            args.moods_bigwig,
            npeaks=args.npeaks,
            max_num_samples=args.max_num_samples,
        )


def attribution_shap_genome(
    meta_file,
    model_config,
    chromosome,
    cell_type,
    output_dir,
    model_base_dir,
    bases_of_interest_list,
    input_seq=None,
    num_background_seqs=100,
):
    """
    Implement Integrated Gradients for the model, providing the background for only the genome
    """
    # tf.compat.v1.disable_v2_behavior()
    # inputs, targets = get_data(meta_file, chromosome, cell_type, output_dir)     # numpy array shape (batch, 1024, 5) (let's set batch = 1)

    # Read model config
    with open(model_config, "r") as f:
        model_config = json.load(f)

    with open("/data/weirauchlab/team/ngun7t/maxatac/scratch/input.npy", "rb") as f:
        inputs = np.load(f)
    with open("/data/weirauchlab/team/ngun7t/maxatac/scratch/target.npy", "rb") as f:
        targets = np.load(f)

    # Load model from dir
    model = load_model_from_dir(model_base_dir, model_config)

    if input_seq is not None:
        input_samples = input_for_interfusion_att(
            input_seq, model_config["INTER_FUSION"]
        )  # This is a list
        if model_config["INTER_FUSION"]:
            contribs_all_samples = []

            for i, input_sample in enumerate(input_samples):
                genome, signal = (
                    input_sample["genome"],
                    input_sample["atac"],
                )  # Shape = (1, seq_len, 4) and (1, seq_len, 1)
                contribs = []
                bases_of_interest = bases_of_interest_list[i]
                for (
                    base_of_interest
                ) in (
                    bases_of_interest
                ):  # example: base_of_interest = [[13], [14], [15]]
                    logging.error(base_of_interest)
                    contrib = run_shap_genome(
                        [genome, signal],
                        model,
                        base_of_interest,
                        num_background_seqs,
                        model_config["INTER_FUSION"],
                    )
                    if type(contrib) == list:
                        contribs.append(contrib[0])

                # This is a list of list with the following structure:
                # [[contribs from 1st sample at nth bin, contribs from 1st sample at (n+1)th bin, ...], [contribs from 2nd sample at nth bin, ...]]
                contribs_all_samples.append(contribs)
            return contribs_all_samples


def attribution_shap_atac(
    meta_file,
    model_config,
    chromosome,
    cell_type,
    output_dir,
    model_base_dir,
    bases_of_interest_list,
    input_seq=None,
    num_background_seqs=100,
):
    """
    Implement Integrated Gradients for the model, providing the background for only the ATAC
    """
    # Read model config
    with open(model_config, "r") as f:
        model_config = json.load(f)

    # Load model from dir
    model = load_model_from_dir(model_base_dir, model_config)

    if input_seq is not None:
        input_samples = input_for_interfusion_att(
            input_seq, model_config["INTER_FUSION"]
        )  # This is a list
        if model_config["INTER_FUSION"]:
            contribs_all_samples = []

            for i, input_sample in enumerate(input_samples):
                genome, signal = (
                    input_sample["genome"],
                    input_sample["atac"],
                )  # Shape = (1, seq_len, 4) and (1, seq_len, 1)
                contribs = []
                bases_of_interest = bases_of_interest_list[i]
                for (
                    base_of_interest
                ) in (
                    bases_of_interest
                ):  # example: base_of_interest = [[13], [14], [15]]
                    logging.error(base_of_interest)
                    contrib = run_shap_atac(
                        [genome, signal],
                        model,
                        base_of_interest,
                        num_background_seqs,
                        model_config["INTER_FUSION"],
                    )
                    if type(contrib) == list:
                        contribs.append(contrib[1])

                # This is a list of list with the following structure:
                # [[contribs from 1st sample at nth bin, contribs from 1st sample at (n+1)th bin, ...], [contribs from 2nd sample at nth bin, ...]]
                contribs_all_samples.append(contribs)
            return contribs_all_samples


def ISM(
    meta_file,
    model_config,
    chromosome,
    cell_type,
    output_dir,
    model_base_dir,
    bases_of_interest_list,
    input_seq=None,
    npeaks=9,
    max_num_samples=15,
    visualize=True,
):
    """
    ISM stands for In-Silico Mutagenesis
    """
    # inputs, targets = get_data(meta_file, chromosome, cell_type, output_dir)     # numpy array shape (batch, 1024, 5) (let's set batch = 1)

    # Currently in debug
    with open("/data/weirauchlab/team/ngun7t/maxatac/scratch/input.npy", "rb") as f:
        inputs = np.load(f)
    with open("/data/weirauchlab/team/ngun7t/maxatac/scratch/target.npy", "rb") as f:
        targets = np.load(f)

    # Read model config
    with open(model_config, "r") as f:
        model_config = json.load(f)

    batch_size = 128
    output_size = 32
    seq_len = 1024
    differences = []

    # Load model from dir
    model = load_model_from_dir(model_base_dir, model_config)

    if input_seq is None:
        # Extract genome sequence and ATAC seq
        target_peaks = np.sum(targets, axis=1)
        relevant_inputs = inputs[target_peaks == npeaks]

        for j in range(min(max_num_samples, relevant_inputs.shape[0])):
            orig_seq = relevant_inputs[j : j + 1, :, :]  # shape (1, 1024, 5)
            difference = run_ism(
                orig_seq,
                model,
                batch_size,
                output_size,
                seq_len,
                model_config["INTER_FUSION"],
            )
            differences.append(difference)
            if visualize:
                output_full_dir = output_dir
                os.makedirs(output_full_dir, exist_ok=True)
                plot_ism_results(
                    difference,
                    save_dir=f"{output_full_dir}/ism-{cell_type}-{chromosome}-{npeaks}peaks-id{j}.png",
                )

    else:
        # input_seq.shape = (num_seq, seq_len, 5)
        num_input_seqs = input_seq.shape[0]
        for j in range(min(num_input_seqs, max_num_samples)):
            bases_of_interest = bases_of_interest_list[j]
            orig_seq = input_seq[j : j + 1, :, :]  # shape (1, 1024, 5)
            difference = run_ism(
                orig_seq,
                model,
                bases_of_interest,
                batch_size,
                output_size,
                seq_len,
                model_config["INTER_FUSION"],
            )  # this is a list
            differences.append(difference)
            if visualize:
                output_full_dir = output_dir
                l = [[str(y) for y in z] for z in bases_of_interest]
                save_dirs = [
                    f"{output_full_dir}/ism-{cell_type}-{chromosome}-bases{'_'.join(b)}-npeaks{npeaks}-id{j}.png"
                    for b in l
                ]
                os.makedirs(output_full_dir, exist_ok=True)
                plot_ism_results(difference, bases_of_interest, save_dirs=save_dirs)

    return differences


def conv_tower_output(
    meta_file,
    model_config,
    chromosome,
    cell_type,
    output_dir,
    model_base_dir,
    max_num_seqs=15,
    technique="pca",
):
    """
    Get the output of the last conv tower layer to see how well this vector representation has already achieved in the classification
    Some methods include PCA - tSNE - UMAP of these representations
    Or building probing classifiers (https://arxiv.org/pdf/2102.12452.pdf)
    """
    # Read model config
    with open(model_config, "r") as f:
        model_config = json.load(f)

    model = load_model_from_dir(model_base_dir, model_config)

    # Get the output of the all the relevant layers (with the name downsampling_conv) and create a sub model
    layer_names = [
        layer.name
        for layer in model.layers
        if "downsampling_conv" in layer.name or "Transformer_block" in layer.name
    ]

    # Get a slice of the data
    # inputs, targets = get_data(meta_file, chromosome, cell_type, output_dir)
    # with open("/data/weirauchlab/team/ngun7t/maxatac/scratch/input.npy", "wb") as f:
    #    np.save(f, inputs)
    # with open("/data/weirauchlab/team/ngun7t/maxatac/scratch/target.npy", "wb") as f:
    #    np.save(f, targets)

    # For debugging, load the data instead
    with open("/data/weirauchlab/team/ngun7t/maxatac/scratch/input.npy", "rb") as f:
        inputs = np.load(f)
    with open("/data/weirauchlab/team/ngun7t/maxatac/scratch/target.npy", "rb") as f:
        targets = np.load(f)

    # Choose inputs that have high target peaks
    target_peaks = np.sum(targets, axis=1)
    num_peaks, num_peak_counts = map(list, np.unique(target_peaks, return_counts=True))
    logging.error(f"Num peaks: {num_peaks} and num peak counts: {num_peak_counts}")

    seqs = []
    true_label_list = []
    window_size = 5

    for out_layer in layer_names:
        sub_model = tf.keras.Model(model.input, model.get_layer(out_layer).output)

        model_outputs = []
        true_labels = []

        # Loop through target sequences based on the number of peaks
        for npeaks, npc in zip(num_peaks, num_peak_counts):
            if npeaks > 0:
                logging.error(f"Num peaks: {npeaks}")
                relevant_inputs = inputs[target_peaks == npeaks]
                relevant_inputs = relevant_inputs[:max_num_seqs]

                # Here we divide sequences based on the number of peaks
                # Instead, a new way is not to cluster sequences based on the count of peaks in the outputs
                # But to cluster subsequences based on whether that corresponding region in the outputs contains a peak
                true_labels.extend([npeaks] * relevant_inputs.shape[0])
                if "downsampling_conv" in out_layer:
                    model_output = sub_model.predict(relevant_inputs)
                else:
                    model_output, _ = sub_model.predict(relevant_inputs)
                model_outputs.append(model_output)

        # I don't know what works best, but first let's average the whole sequence and cluster them
        seq_embeddings = []
        for model_output in model_outputs:
            seq_avg = np.mean(model_output, axis=1)
            seq_embeddings.append(seq_avg)

        seq_embeddings = np.concatenate(seq_embeddings, axis=0)
        logging.error(f"Seq_embeddings.shape = {seq_embeddings.shape}")
        true_labels = np.array(true_labels)
        logging.error(f"True_labels.shape = {true_labels.shape}")

        # PCA or cosine similarity clustering
        if technique == "pca":
            pca = PCA(n_components=2)
            seq = pca.fit_transform(seq_embeddings)
        elif technique == "tsne":
            tsne = TSNE(n_components=2)
            seq = tsne.fit_transform(seq_embeddings)

        seqs.append(seq)
        true_label_list.append(true_labels)

    # Plot
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(25, 25))
    axes = axes.ravel()

    for i in range(len(layer_names)):
        seq, true_labels = seqs[i], true_label_list[i]
        for val in num_peaks:
            seq_with_label = seq[true_labels == val]
            axes[i].scatter(seq_with_label[:, 0], seq_with_label[:, 1], label=val)
            axes[i].set_title(layer_names[i])

        box = axes[i].get_position()
        axes[i].set_position([box.x0, box.y0, box.width * 0.8, box.height])
        axes[i].legend(loc="center left", bbox_to_anchor=(1, 0.5))

    plt.tight_layout()
    plt.savefig(f"/data/weirauchlab/team/ngun7t/maxatac/scratch/{technique}_layers.png")


def check_attention_offset_distribution(
    meta_file, model_config, chromosome, cell_type, output_dir, model_base_dir
):
    """
    Concept: visualize the distribution of the attention weights further away from the base of interest
    """
    batch_size = 100
    # Read model config
    with open(model_config, "r") as f:
        model_config = json.load(f)

    model = load_model_from_dir(model_base_dir, model_config)

    # Get the output of the all the relevant layers (with the name downsampling_conv) and create a sub model
    layer_names = [
        layer.name for layer in model.layers if "Transformer_block" in layer.name
    ]
    sub_models = []

    # Get all the necessary layers
    for layer_name in layer_names:
        sub_model = tf.keras.Model(model.inputs, model.get_layer(layer_name).output)
        sub_models.append(sub_model)

    # Get the data (for debugging, we use the precomputed data)
    with open("/data/weirauchlab/team/ngun7t/maxatac/scratch/input.npy", "rb") as f:
        inputs = np.load(f)
    with open("/data/weirauchlab/team/ngun7t/maxatac/scratch/target.npy", "rb") as f:
        targets = np.load(f)

    random_inds = np.random.choice(inputs.shape[0], size=(batch_size,), replace=False)
    inputs = inputs[random_inds]

    # Get the weights
    base_attentions = []
    for sub_model in sub_models:
        outputs = sub_model.predict(inputs)  # weights.shape = (batch_size, 4, 512, 512)
        if len(outputs) == 2:
            _, weights = outputs
        elif len(outputs) == 3:
            _, weights, _ = outputs
        base_highest_attention = np.argmax(
            weights, axis=2
        )  # shape = (batch_size, 4, 512)
        base_highest_att_med = np.median(
            base_highest_attention, axis=0
        )  # shape = (4, 512)
        dist = (
            np.repeat(
                np.expand_dims(
                    np.arange(base_highest_att_med.shape[1], dtype=float), axis=0
                ),
                repeats=base_highest_att_med.shape[0],
                axis=0,
            )
            - base_highest_att_med
        )
        base_attentions.append(dist)

    # Draw a scatter plot
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(30, 30))
    axes = axes.ravel()
    x = list(range(512))
    colors = ["blue", "red", "green", "black"]

    for i, base_att in enumerate(base_attentions):
        for head in range(base_att.shape[0]):
            axes[i].plot(
                x, base_att[head], "o", label=f"Head {head}", color=colors[head]
            )

        axes[i].legend()
        axes[i].set_title(f"Layer {i}")

    plt.tight_layout()
    plt.savefig(f"/data/weirauchlab/team/ngun7t/maxatac/scratch/attention_bases.png")


def check_layer_contributions(
    meta_file,
    model_config,
    chromosome,
    cell_type,
    output_dir,
    model_base_dir,
    max_num_seqs=15,
    npeaks=9,
    mode="b",
):
    """
    Check whether each transformer layer plays an important role in the predictions
    To do it, extract the conv layer after the final transformer layer,
    then sequentially extract the transformer layer outputs and feed to the conv layer
    and measure the difference between the original output and the new output

    Another way is to remove one transformer layer at the time and see what happens

    This function is only currently designed not for interfusion models, RPE models that have
    transformer layers with names Transformer_block, and models with the same output conv layers

    There's a lot of hard-coded things in this function
    """
    # Read model config
    with open(model_config, "r") as f:
        model_config = json.load(f)

    model = load_model_from_dir(model_base_dir, model_config)

    # Get the output of the all the transformer layers and the last downsampling_conv and create a sub model
    relevant_layers = [
        layer.name for layer in model.layers if "Transformer_block" in layer.name
    ]
    last_downsampling_conv_layer = sorted(
        [layer.name for layer in model.layers if "downsampling_conv" in layer.name]
    )[-1]
    relevant_layers.append(last_downsampling_conv_layer)

    # Get the part of the model after the last transformer layer
    output_model = tf.keras.Model(
        model.get_layer("conv1d_1").input, model.layers[-1].output
    )

    # For debugging, load the data instead
    with open("/data/weirauchlab/team/ngun7t/maxatac/scratch/input.npy", "rb") as f:
        inputs = np.load(f)
    with open("/data/weirauchlab/team/ngun7t/maxatac/scratch/target.npy", "rb") as f:
        targets = np.load(f)

    target_peaks = np.sum(targets, axis=1)
    relevant_inputs = inputs[target_peaks == npeaks]
    relevant_inputs = relevant_inputs[:max_num_seqs]

    orig_output = model.predict(relevant_inputs)  # should have shape (batch_size, 32)
    output_diff = []

    # Sequentially pass the transformer layer outputs to the last layer
    if mode == "a":
        for layer in relevant_layers:
            logging.error(layer)
            sub_model = tf.keras.Model(model.input, model.get_layer(layer).output)

            # Make a forward pass through both models
            if "downsampling_conv" in layer:
                trans_layer_output = sub_model.predict(relevant_inputs)
            else:
                outputs = sub_model.predict(relevant_inputs)
                if len(outputs) == 2:
                    trans_layer_output, _ = outputs
                elif len(outputs) == 3:
                    trans_layer_output, _, _ = outputs
            final_output = output_model.predict(trans_layer_output)
            output_diff.append(np.sum(np.abs(final_output - orig_output), axis=1))

        # Draw the plots probably using box plots
        plt.figure()
        sns.catplot(data=output_diff)
        plt.xticks(np.arange(len(relevant_layers)) + 0.5, relevant_layers, rotation=90)
        plt.title(f"Diff in output of transformer layers, peak counts = {npeaks}")
        plt.tight_layout()
        plt.savefig(f"/data/weirauchlab/team/ngun7t/maxatac/scratch/layer_diff.png")

    if mode == "b":
        # Remove a transformer layer one at a time
        transformer_layers = [
            l for l in relevant_layers if "downsampling_conv" not in l
        ]
        for i in range(len(transformer_layers)):
            first_output = transformer_layers[i - 1]
            if i == 0:
                first_output = last_downsampling_conv_layer
                last_input = transformer_layers[i + 1]
            elif i == len(transformer_layers) - 1:
                last_input = "conv1d_1"
            else:
                last_input = transformer_layers[i + 1]

            first_model = tf.keras.Model(
                model.input, model.get_layer(first_output).output
            )
            last_model = tf.keras.Model(
                model.get_layer(last_input).input, model.layers[-1].output
            )
            temp = first_model.predict(relevant_inputs)
            if type(temp) == tuple:
                temp = temp[0]
            final_output = last_model.predict(temp)
            output_diff.append(np.sum(np.abs(final_output - orig_output), axis=1))

        # Draw the plots probably using box plots
        plt.figure(figsize=(7, 7))
        sns.catplot(data=output_diff)
        plt.xticks(np.arange(len(transformer_layers)), transformer_layers, rotation=90)
        plt.title(f"Diff in output of when missing layers, peak counts = {npeaks}")
        plt.tight_layout()
        plt.savefig(f"/data/weirauchlab/team/ngun7t/maxatac/scratch/layer_diff_b.png")


def plot_attention_weights(
    meta_file,
    model_config,
    chromosome,
    cell_type,
    output_dir,
    model_base_dir,
    max_num_samples=10,
    npeaks=9,
    input_seq=None,
    num_heads=4,
    aggregation_func="mean",
    z_score=True,
    visualize=True,
):
    """
    Plot the attention weights of all heads for all transformer layers when feeding with
    one sample input sequence
    The rightmost column is the aggregated attention weights across heads,
    and the bottom most row is the aggregated attention weights across layers
    Aggregation function can be mean or max
    z-score att weights??
    """
    # Read model config
    with open(model_config, "r") as f:
        model_config = json.load(f)
    model = load_model_from_dir(model_base_dir, model_config)
    if aggregation_func == "mean":
        func = np.mean
    elif aggregation_func == "max":
        func = np.amax

    if input_seq is not None:
        input_seq = input_seq[:max_num_samples, :, :]
        logging.error(input_seq.shape)
        input_sample = input_for_interfusion_att(
            input_seq, model_config["INTER_FUSION"]
        )
    else:
        # For debugging, load the data instead
        # inputs, targets = get_data(meta_file, chromosome, cell_type, output_dir)
        with open("/data/weirauchlab/team/ngun7t/maxatac/scratch/input.npy", "rb") as f:
            inputs = np.load(f)
        with open(
            "/data/weirauchlab/team/ngun7t/maxatac/scratch/target.npy", "rb"
        ) as f:
            targets = np.load(f)
        input_sample = input_for_interfusion_att(
            inputs[:max_num_samples, :, :], model_config["INTER_FUSION"]
        )

    relevant_layers = [
        layer.name for layer in model.layers if "Transformer_block" in layer.name
    ]
    num_layers = len(relevant_layers)

    # Get all the attention weights for all sequences
    att_matrices = []
    for s in range(min(len(input_sample), max_num_samples)):
        input_sequence = input_sample[s]
        att_layers = []  # Store attention weights across layers
        for l in range(num_layers):
            # Build a submodel
            transformer_name = relevant_layers[l]
            submodel = tf.keras.Model(
                model.inputs, model.get_layer(transformer_name).output
            )

            # Make an inference
            outputs = submodel.predict(
                input_sequence, verbose=0
            )  # att_weights.shape = (num_head, height, width)
            if len(outputs) == 2:
                _, att_weights = outputs
            elif len(outputs) == 3:
                _, att_weights, _ = outputs
            seq_len = att_weights.shape[-1]
            att_weights = tf.reshape(att_weights, (-1, seq_len, seq_len))
            att_layers.append(att_weights)

        att_matrix = np.stack(
            att_layers, axis=0
        )  # shape = (num_layer, num_head, height, width)
        if z_score:
            att_matrix = zscore(att_matrix)
        att_matrices.append(att_matrix)

    # If z-score:
    if z_score:
        palette = sns.diverging_palette(214, 20, s=200, l=50, as_cmap=True)
    else:
        palette = None

    if visualize:
        for s, att_matrix in enumerate(att_matrices):
            # Plot all the attention weights for all sequences
            fig, axes = plt.subplots(
                nrows=num_layers + 1, ncols=num_heads + 1, figsize=(40, 40)
            )
            for l in range(num_layers):
                for h in range(num_heads):
                    sns.heatmap(
                        data=att_matrix[l, h, :, :],
                        ax=axes[l][h],
                        cmap=palette,
                        center=0,
                    )
                    axes[l][h].set_title(f"L{l+1} - H{h+1}")
                    axes[l][h].set_axis_off()

                # Plot the aggregated values
                agg_weights_across_heads = func(att_matrix, axis=1)[l]
                sns.heatmap(
                    data=agg_weights_across_heads,
                    ax=axes[l][num_heads],
                    cmap=palette,
                    center=0,
                )
                axes[l][num_heads].set_title(f"Aggregated L{l+1}")
                axes[l][num_heads].set_axis_off()
                agg_weights_across_layers = func(att_matrix, axis=0)[l]
                sns.heatmap(
                    data=agg_weights_across_layers,
                    ax=axes[num_layers][l],
                    cmap=palette,
                    center=0,
                )
                axes[num_layers][l].set_title(f"Aggregated H{l+1}")
                axes[num_layers][l].set_axis_off()

            # Final aggregated weights
            agg_all_weights = func(att_matrix, axis=(0, 1))
            sns.heatmap(
                data=agg_all_weights,
                ax=axes[num_layers][num_heads],
                cmap=palette,
                center=0,
            )
            axes[num_layers][num_heads].set_title(f"Aggregated all")
            axes[num_layers][num_heads].set_axis_off()

            plt.tight_layout()
            plt.savefig(
                f"{output_dir}/attention-{cell_type}-{chromosome}-{aggregation_func}-npeaks{npeaks}-id{s}.png"
            )

    return att_matrices


def plot_latent_signal_vector(
    meta_file,
    model_config,
    chromosome,
    cell_type,
    output_dir,
    model_base_dir,
    input_seq=None,
    visualize=False,
    max_num_samples=15,
    npeaks=9,
):
    """
    Plot the latent ATAC-seq signal at multiple layers to MHA
        - At the last layer of the conv tower
        - At the last layer before feeding to the transformer
        - At the layer after layer norm inside MHA
    """
    # Read model config
    with open(model_config, "r") as f:
        model_config = json.load(f)

    model = load_model_from_dir(model_base_dir, model_config)

    # Get the name of all relevant layers
    all_layer_names = get_intermediate_layer_names(
        model_base_dir, model_config, branch="atac"
    )

    if input_seq is not None:
        input_seq = input_seq[:max_num_samples, :, :]
        logging.error(input_seq.shape)
        input_sample = input_for_interfusion_ism(
            input_seq, model_config["INTER_FUSION"]
        )
        latent_vectors = []

        layers_before_transformer = [
            l for l in all_layer_names if "Transformer_block" not in l
        ]
        for layer in layers_before_transformer:
            layer_output = tf.keras.Model(model.inputs, model.get_layer(layer).output)(
                input_sample
            )  # Shape = (num_samples, seq_len, feature_dim)
            # Very hard-coded here
            if layer_output.shape[1] != 256:
                if "fusion" in layer:
                    latent_vectors.append(layer_output[:, 256:, :])
                elif "maxpool" in layer:
                    latent_vectors.append(
                        np.amax(
                            np.reshape(
                                layer_output,
                                (
                                    layer_output.shape[0],
                                    layer_output.shape[1] // 2,
                                    2,
                                    layer_output.shape[2],
                                ),
                            ),
                            axis=2,
                        )
                    )
            else:
                latent_vectors.append(layer_output)
        transformer_names = [l for l in all_layer_names if "Transformer_block" in l]
        # Latent vector inside MHA after first layer norm and before MHA module
        for t in transformer_names:
            _, _, transformer_latent_vectors = tf.keras.Model(
                model.inputs, model.get_layer(t).output
            )(input_sample)
            latent_vectors.append(transformer_latent_vectors[0][:, 256:, :])

        # Convert latent_vectors to a numpy array of shape (num_samples, num_types_of_layer, seq_len, feature_dim)
        latent_vectors = np.swapaxes(np.array(latent_vectors), 0, 1)

        if visualize:
            visualize_latent_vectors(
                latent_vectors,
                all_layer_names,
                f"{output_dir}/latent_atac-{cell_type}-{chromosome}-npeaks{npeaks}",
            )

        return latent_vectors, all_layer_names


def plot_latent_genome_vector(
    meta_file,
    model_config,
    chromosome,
    cell_type,
    output_dir,
    model_base_dir,
    input_seq=None,
    visualize=False,
    max_num_samples=15,
    npeaks=9,
):
    """
    Plot the latent ATAC-seq signal at multiple layers to MHA
        - At the last layer of the conv tower
        - At the last layer before feeding to the transformer
        - At the layer after layer norm inside MHA
    """
    # Read model config
    with open(model_config, "r") as f:
        model_config = json.load(f)

    model = load_model_from_dir(model_base_dir, model_config)

    # Get the name of all relevant layers
    all_layer_names = get_intermediate_layer_names(
        model_base_dir, model_config, branch="genome"
    )

    if input_seq is not None:
        input_seq = input_seq[:max_num_samples, :, :]
        logging.error(input_seq.shape)
        input_sample = input_for_interfusion_ism(
            input_seq, model_config["INTER_FUSION"]
        )
        latent_vectors = []

        # Conv layers
        layers_before_transformer = [
            l for l in all_layer_names if "Transformer_block" not in l
        ]
        for layer in layers_before_transformer:
            layer_output = tf.keras.Model(model.inputs, model.get_layer(layer).output)(
                input_sample
            )  # Shape = (num_samples, seq_len, feature_dim)
            if layer_output.shape[1] != 256:  # hard-coded
                if "fusion" in layer:
                    latent_vectors.append(layer_output[:, :256, :])
                elif "maxpool" in layer:
                    latent_vectors.append(
                        np.amax(
                            np.reshape(
                                layer_output,
                                (
                                    layer_output.shape[0],
                                    layer_output.shape[1] // 2,
                                    2,
                                    layer_output.shape[2],
                                ),
                            ),
                            axis=2,
                        )
                    )
            else:
                latent_vectors.append(layer_output)

        # Latent vector inside MHA after first layer norm and before MHA module
        transformer_names = [l for l in all_layer_names if "Transformer_block" in l]
        for t in transformer_names:
            _, _, transformer_latent_vectors = tf.keras.Model(
                model.inputs, model.get_layer(t).output
            )(input_sample)
            latent_vectors.append(
                transformer_latent_vectors[0][:, :256, :]
            )  # Hard-coded

        # Convert latent_vectors to a numpy array of shape (num_samples, num_types_of_layer, seq_len, feature_dim)
        latent_vectors = np.swapaxes(np.array(latent_vectors), 0, 1)

        if visualize:
            visualize_latent_vectors(
                latent_vectors,
                all_layer_names,
                f"{output_dir}/latent_genome-{cell_type}-{chromosome}-npeaks{npeaks}",
            )

        return latent_vectors, all_layer_names


def ism_attention_weights_correlation(
    meta_file,
    model_config,
    chromosome,
    cell_type,
    output_dir,
    model_base_dir,
    moods_bigwig,
    npeaks=9,
    max_num_samples=20,
):
    """
    Correlation between the base from ISM with the highest change, and the attention rows of that base of the attention weights
    A caveat is that the attention weights are not at the same dimension as the original input sequence, so a direct mapping is not possible
    There are several approaches:
        - Find the highest base location in ISM, then find the highest base in the attention weights, then make a scatterplot for measuring R^2 coeff
        - Plot the attention rows similar to the supp figure of Enformer paper
        -
    """
    # Read model config
    with open(model_config, "r") as f:
        model_config_dict = json.load(f)

    # Plot attention row
    def plot_att_row(att_row, ax):
        """
        att_row can have shape (num_layer, num_head, row_len)
        """
        num_layer, num_head = att_row.shape[0], att_row.shape[1]
        att_row_resized = np.reshape(att_row, (-1, att_row.shape[-1]))
        sns.heatmap(att_row_resized, ax=ax)
        yticks_loc = np.arange(0, num_head * num_layer, 2) + 0.5
        yticks_label = [
            f"L{l}-H{h}" for l in range(0, num_layer) for h in range(0, num_head, 2)
        ]
        ax.set_yticks(yticks_loc, yticks_label, rotation=0)

    # Find location of longest peak
    def longest_peak(arr):
        ans, temp, loc = 1, 1, 0
        for i in range(1, len(arr)):
            if arr[i] == arr[i - 1]:
                temp += 1
            else:
                ans = max(ans, temp)
                loc = i - ans
                temp = 1
            ans = max(ans, temp)
        ans = max(ans, temp)
        return loc, loc + ans

    visualize = True
    agg_func = "max"
    num_bg_shap = 100
    model_name = ntpath.basename(model_base_dir)

    # Create the dirs:
    output_dir = f"{output_dir}/{model_name}/{cell_type}/npeaks{npeaks}"
    os.makedirs(output_dir, exist_ok=True)

    # Load the input sequences
    # moods.shape = (num_samples, seq_len)
    inputs, targets, moods = get_data(
        meta_file, chromosome, cell_type, output_dir, moods_bigwig=moods_bigwig
    )
    moods = np.nan_to_num(moods)
    moods[moods > 0] = 1

    with open(f"{'/'.join(output_dir.split('/')[:-1])}/inputs.npy", "wb") as f:
        np.save(f, inputs)
    with open(f"{'/'.join(output_dir.split('/')[:-1])}/targets.npy", "wb") as f:
        np.save(f, targets)
    with open(f"{'/'.join(output_dir.split('/')[:-1])}/moods.npy", "wb") as f:
        np.save(f, moods)

    # Choose inputs that have high target peaks
    target_peaks = np.sum(targets, axis=1)
    relevant_inputs = inputs[target_peaks == npeaks]
    relevant_outputs = targets[target_peaks == npeaks]
    relevant_moods = moods[target_peaks == npeaks]

    input_seq = relevant_inputs[:max_num_samples, :, :]
    output_seq = np.squeeze(
        relevant_outputs[:max_num_samples, :]
    )  # shape = (num_samples, 32)
    moods_seq = relevant_moods[0:max_num_samples, :]
    logging.error(input_seq.shape)

    # Define the list of bases of interest
    # Define the location of the center of the longest true peak
    central_peaks, bases_of_interest_list = [], []
    for sample in range(output_seq.shape[0]):
        start, end = longest_peak(list(output_seq[sample]))
        avg = (start + end) // 2
        bases_of_interest = [[13], [14], [15], [16]]
        if avg not in [13, 14, 15, 16]:
            bases_of_interest.append([avg])
        bases_of_interest_list.append(bases_of_interest)

    # Then run ATAC-seq latent vector, atac_latent_vectors.shape = (num_samples, num_types_of_layer, seq_len, feature_dims)
    atac_latent_vectors, all_layer_names = plot_latent_signal_vector(
        meta_file,
        model_config,
        chromosome,
        cell_type,
        output_dir,
        model_base_dir,
        input_seq=input_seq,
        max_num_samples=max_num_samples,
        npeaks=npeaks,
        visualize=visualize,
    )

    # Then run genome latent vector, genome_latent_vectors.shape = (num_samples, num_types_of_layer, seq_len, feature_dims)
    genome_latent_vectors, all_layer_names = plot_latent_genome_vector(
        meta_file,
        model_config,
        chromosome,
        cell_type,
        output_dir,
        model_base_dir,
        input_seq=input_seq,
        max_num_samples=max_num_samples,
        npeaks=npeaks,
        visualize=visualize,
    )

    # Then run attribution SHAP for ATAC only
    # This is a list of list, the outer list contains the results for each sample, the inner list contains results for different outputs
    shap_atac_results = attribution_shap_atac(
        meta_file,
        model_config,
        chromosome,
        cell_type,
        output_dir,
        model_base_dir,
        bases_of_interest_list,
        input_seq=input_seq,
        num_background_seqs=num_bg_shap,
    )

    # Then run attribution SHAP
    # This is a list of list, the outer list contains the results for each sample, the inner list contains results for different outputs
    shap_genome_results = attribution_shap_genome(
        meta_file,
        model_config,
        chromosome,
        cell_type,
        output_dir,
        model_base_dir,
        bases_of_interest_list,
        input_seq=input_seq,
        num_background_seqs=num_bg_shap,
    )

    # First run ISM and get the output vector
    # This is a list of list
    ism_results = ISM(
        meta_file,
        model_config,
        chromosome,
        cell_type,
        output_dir,
        model_base_dir,
        bases_of_interest_list,
        input_seq=input_seq,
        max_num_samples=max_num_samples,
        visualize=visualize,
        npeaks=npeaks,
    )

    # Then run attention weights
    att_matrices = plot_attention_weights(
        meta_file,
        model_config,
        chromosome,
        cell_type,
        output_dir,
        model_base_dir,
        input_seq=input_seq,
        visualize=visualize,
        z_score=False,
        aggregation_func=agg_func,
        max_num_samples=max_num_samples,
        npeaks=npeaks,
    )  # shape = (num_layer, num_head, height, width)

    # Then resize the output match with the dimension of the attention weights
    output_seq = np.repeat(
        output_seq,
        repeats=((att_matrices[0].shape[-1] // 2) // output_seq.shape[-1]),
        axis=1,
    )

    assert len(ism_results) == len(
        att_matrices
    ), f"The number of attention matrices={len(att_matrices)} and ISM={len(ism_results)} results are not equal!"
    for s in range(min(max_num_samples, relevant_inputs.shape[0])):
        ism_result = ism_results[s]  # ism_result is a list of ism vectors for each bin
        att_matrix = att_matrices[s]
        atac_latent_vector = atac_latent_vectors[s, ...]
        genome_latent_vector = genome_latent_vectors[s, ...]
        shap_genome_result = shap_genome_results[s]  # type(shap_genome_result) = list
        shap_atac_result = shap_atac_results[s]
        relevant_output_seq = np.expand_dims(output_seq[s], axis=0)
        moods_sample = np.expand_dims(moods_seq[s], axis=0)  # shape = (1,1024)

        input_length = ism_result[0].shape[0]
        num_mha, num_head, att_weight_width = (
            att_matrix.shape[0],
            att_matrix.shape[1],
            att_matrix.shape[-1],
        )

        # Reshape input ATAC-seq signal
        signal = relevant_inputs[s, :, 4:5]
        signal_resized = np.amax(
            np.reshape(
                np.amax(signal, axis=1), (-1, (input_length // att_weight_width) * 2)
            ),
            axis=-1,
        )

        # Reshape ISM vector
        ism_resized = np.amax(
            np.reshape(
                np.amax(ism_result[0], axis=1),
                (-1, (input_length // att_weight_width) * 2),
            ),
            axis=-1,
        )

        # Identify the location of the base with the highest change in ISM
        ism_highest_val_base = np.argmax(ism_resized)

        # Visualize the attention row at the highest base
        # When doing interfusion, there are two rows that correspond to the original base (genome branch and ATAC branch)
        if model_config_dict["INTER_FUSION"]:
            fig, axes = plt.subplots(
                nrows=22,
                ncols=1,
                figsize=(17, 30),
                sharex=True,
                constrained_layout=True,
            )
            first_row_ind = ism_highest_val_base
            second_row_ind = (att_weight_width // 2) + first_row_ind

            genome_genome = att_matrix[:, :, first_row_ind, : att_weight_width // 2]
            genome_atac = att_matrix[:, :, first_row_ind, att_weight_width // 2 :]
            atac_genome = att_matrix[:, :, second_row_ind, : att_weight_width // 2]
            atac_atac = att_matrix[:, :, second_row_ind, att_weight_width // 2 :]

            # Plot the true ATAC-seq signal
            axes[0].plot(signal_resized)
            axes[0].set_title("ATAC_seq")

            # Plot the true output (axes 1)
            sns.heatmap(relevant_output_seq, ax=axes[1])
            axes[1].set_title("True outputs")

            # Plot the MOODS output (axes 2)
            sns.heatmap(moods_sample, ax=axes[2])
            axes[2].set_title("MOODS")

            # Plot the ATAC-seq latent signals (axes 3-8)
            for i in range(atac_latent_vector.shape[0]):
                latent_atac = atac_latent_vector[
                    i, ...
                ]  # shape = (seq_len, feature_dims)
                if latent_atac.shape[0] == 512:  # very hard code value here
                    latent_atac = latent_atac[latent_atac.shape[0] // 2 :, ...]
                axes[i + 3].plot(np.mean(latent_atac, axis=1), "-r")
                axes[i + 3].fill_between(
                    np.arange(latent_atac.shape[0]),
                    np.mean(latent_atac, axis=1) + np.std(latent_atac, axis=1),
                    np.mean(latent_atac, axis=1) - np.std(latent_atac, axis=1),
                    color="#9DD9F3",
                )
                # axes[i+3].plot(np.amax(latent_atac, axis=1), "--g")
                # axes[i+3].plot(np.amin(latent_atac, axis=1), "--g")
                axes[i + 3].set_title(
                    f"Latent ATAC-seq vector after {all_layer_names[i]}"
                )

            # Plot the genome latent signal (axes 9-14)
            for i in range(genome_latent_vector.shape[0]):
                latent_genome = genome_latent_vector[
                    i, ...
                ]  # shape = (seq_len, feature_dims)
                if latent_genome.shape[0] == 512:  # very hard code value here
                    latent_genome = latent_genome[latent_genome.shape[0] // 2 :, ...]
                axes[i + 9].plot(np.mean(latent_genome, axis=1), "-r")
                axes[i + 9].fill_between(
                    np.arange(latent_genome.shape[0]),
                    np.mean(latent_genome, axis=1) + np.std(latent_genome, axis=1),
                    np.mean(latent_genome, axis=1) - np.std(latent_genome, axis=1),
                    color="#9DD9F3",
                )
                # axes[i+9].plot(np.amax(latent_genome, axis=1), "--g")
                # axes[i+9].plot(np.amin(latent_genome, axis=1), "--g")
                axes[i + 9].set_title(
                    f"Latent genome vector after {all_layer_names[i]}"
                )

            # Plot ISM (axes 15)
            l = [[str(y) for y in z] for z in bases_of_interest_list[s]]
            for k, ism in enumerate(ism_result):
                ism_resized = np.amax(
                    np.reshape(
                        np.amax(ism, axis=1),
                        (-1, (input_length // att_weight_width) * 2),
                    ),
                    axis=-1,
                )
                axes[15].plot(ism_resized, label=f"Bin {'-'.join(l[k])}")
                axes[15].legend()
            axes[15].set_title("ISM")

            # Plot IG atac(axes 16)
            for k, shap in enumerate(shap_atac_result):
                logging.error(shap.shape)  # shap has shape (1, seq_len, 1)
                shap_resized = np.amax(
                    np.reshape(shap, (-1, (input_length // att_weight_width) * 2)),
                    axis=-1,
                )
                axes[16].plot(shap_resized, label=f"Bin {'-'.join(l[k])}")
                axes[16].legend()
            axes[16].set_title("Integrated Gradients on only ATAC-seq")

            # Plot IG genome (axes 17)
            for k, shap in enumerate(shap_genome_result):
                logging.error(shap.shape)  # shap has shape (1, seq_len, 4)
                shap_resized = np.amax(
                    np.reshape(
                        np.amax(np.squeeze(shap), axis=1),
                        (-1, (input_length // att_weight_width) * 2),
                    ),
                    axis=-1,
                )
                axes[17].plot(shap_resized, label=f"Bin {'-'.join(l[k])}")
                axes[17].legend()
            axes[17].set_title("Integrated Gradients on only genome")

            # Plot attention (axes 18-21)
            plot_att_row(genome_genome, ax=axes[18])
            axes[18].set_title(f"genome - genome at bin [15]")
            plot_att_row(genome_atac, ax=axes[19])
            axes[19].set_title(f"genome - atac at bin [15]")
            plot_att_row(atac_genome, ax=axes[20])
            axes[20].set_title(f"atac - genome at bin [15]")
            plot_att_row(atac_atac, ax=axes[21])
            axes[21].set_title(f"atac - atac at bin [15]")

            # plt.tight_layout()
            plt.savefig(
                f"{output_dir}/z-ismmat-{model_name}-{cell_type}-npeaks{npeaks}-id{s}.png"
            )


def input_maximization(
    meta_file,
    chromosome,
    cell_type,
    output_dir,
    model_base_dir,
    npeaks=9,
    max_num_samples=2,
):
    """
    This function aims to maximize the inputs to force the model to generate the output that best
    resembles in the true output (using cross entropy loss)
    It can be either given a (sequence+signal) pair, keep the sequence constant, and backprop on
    the signal, or vice versa.
    Or even backprop on both the sequence and the signal and observe the results
    """
    # Load the input sequences
    with open("/data/weirauchlab/team/ngun7t/maxatac/scratch/input.npy", "rb") as f:
        inputs = np.load(f)
    with open("/data/weirauchlab/team/ngun7t/maxatac/scratch/target.npy", "rb") as f:
        targets = np.load(f)

    # Choose inputs that have high target peaks
    target_peaks = np.sum(targets, axis=1)
    relevant_inputs = inputs[target_peaks == npeaks]
    relevant_outputs = targets[target_peaks == npeaks]
    relevant_inputs = relevant_inputs[:max_num_samples, :, :]
    relevant_outputs = relevant_outputs[:max_num_samples, :, :]

    # Run input atac maximization
    optimized_signal = input_maximization_atac_interfusion(
        relevant_inputs, relevant_outputs, model_base_dir
    )  # shape = (num_samples, seq_len, 1)

    # Plot the true signal vs the optimized signal
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 7))


def check_attention_on_base():
    """
    Check attention of other locations on a particular location
    Note that these locations do not directly map to the original base, as they have passed
    through a positional encoding layer
    """


def check_attention_correlation():
    """
    Run the experiments for checking the attention weights across the best performing models
    and across the best epochs within the same model
    """
    #
