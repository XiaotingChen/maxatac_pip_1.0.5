from maxatac.utilities.system_tools import replace_extension, remove_tags, Mute
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import logomaker as lm
import pyBigWig

import tensorflow as tf

with Mute():
    from tensorflow.keras.utils import plot_model
    from maxatac.utilities.transformer_interpret_tools import one_hot_to_tokens



def plot_attention_weights(model, transformer_names, data_sample, num_heads, file_location, use_rpe):
    """
    Plot the attention weights of all heads for all transformer layers
    data_sample has shape (1, 1024, 5), 1 is the batch size and we want to keep it as 1
    """
    num_layers = len(transformer_names)
    fig, axes = plt.subplots(nrows=num_layers, ncols=num_heads, figsize=(40, 40))

    for l in range(num_layers):

        # Build a submodel
        transformer_name = transformer_names[l]
        submodel = tf.keras.Model(model.inputs, model.get_layer(transformer_name).output)

        # Make an inference
        if use_rpe:
            _, att_weights = submodel.predict(data_sample, verbose=0)
        else:
            att_weights = submodel.predict(data_sample, verbose=0)
        seq_len = att_weights.shape[-1]
        att_weights = tf.reshape(att_weights, (-1, seq_len, seq_len))
        
        for h in range(num_heads):
            att_weights_per_head = att_weights[h]
            sns.heatmap(data=att_weights_per_head, ax=axes[l][h])
            axes[l][h].set_title(f"L{l+1} - H{h+1}")
            axes[l][h].set_axis_off()

    plt.tight_layout()
    plt.savefig(
        os.path.join(file_location, "attention_weights.png"),
    )


def plot_ism_results(superseq, bases_of_interest, save_dirs):
    """
    Plot ISM results
    seq has shape of seq_lenx4
    The one hot encoded rule is ACGT
    """
    bases = ["A", "C", "G", "T"]
    for i, base_of_interest in enumerate(bases_of_interest):
        seq = superseq[i]
        base_dict = {
            b: seq[:, i] for i, b in enumerate(bases)
        }

        # Convert base_dict into a pd Dataframe for logo sequence
        df = pd.DataFrame.from_dict(base_dict)

        # Get the changes across the base
        max_change = np.max(seq, axis=1)
        min_change = np.min(seq, axis=1)
        
        # Plot stuff        
        fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(35, 8), sharex=True, constrained_layout=True)

        # The line plot
        ax[0].plot(max_change, label="Max positive change")
        ax[0].plot(min_change, label="Min negative change")
        ax[0].set_ylabel("Expression difference")
        ax[0].legend()

        # The heatmap
        sns.heatmap(seq.T, ax=ax[1])
        ax[1].set_title("Expression difference when changing base")
        ax[1].set_xlabel("Base ind")
        ax[1].set_ylabel("Expression difference")
        #ax[1].set_xticks(list(range(len(orig_seq)-start_ind-end_ind)), list(range(start_ind, len(orig_seq)-end_ind)))
        ax[1].set_yticks([0, 1, 2, 3], bases)

        # The logo
        lm.Logo(df, font_name = 'Arial Rounded MT Bold', ax=ax[2])

        plt.savefig(save_dirs[i])


def export_model_structure(model, file_location, suffix="_model_structure", ext=".png", skip_tags="_{epoch}"):
    plot_model(
        model=model,
        show_shapes=True,
        show_layer_names=True,
        to_file=replace_extension(
            remove_tags(file_location, skip_tags),
            suffix + ext
        )
    )


def export_binary_metrics(history, tf, RR, ARC, file_location, best_epoch, suffix="_model_dice_acc", ext=".png",
                          style="seaborn-whitegrid", log_base=10, skip_tags="_{epoch}"):
    plt.style.use(style)
    fig, ((ax1, ax2)) = plt.subplots(1, 2, sharex=False, sharey=False, figsize=(18, 6), dpi=200)
    fig.suptitle('Loss and Dice Coefficient with random ratio set at ' + str(RR) + '  for ' + tf + ' using the ' + ARC + ' architecture', fontsize=22)

    ### Loss
    t_y = history.history['loss']
    v_y = history.history["val_loss"]

    x = [int(i) for i in range(1, len(t_y) + 1)]

    ax1.plot(x, t_y, marker='o', lw=3)
    ax1.plot(x, v_y, marker='o', lw=3)

    ax1.set_title("Model Loss", size=20)
    ax1.set_ylabel("Loss", size=20)
    ax1.set_xlabel("Epoch", size=20)

    ax1.legend(["Training", "Validation"], loc="upper right")

    ax1.set_xticks(x)
    ax1.set_xlim(0, )
    ax1.set_ylim(0, )
    ax1.tick_params(labelsize=16)
    ax1.axvline(best_epoch, lw=3, color="red")

    ax1.tick_params(axis="y", labelsize=16)
    ax1.tick_params(axis="x", labelsize=12, rotation=90)

    ### Dice Coefficient
    t_y = history.history['dice_coef']
    v_y = history.history["val_dice_coef"]

    ax2.plot(x, t_y, marker='o', lw=3)
    ax2.plot(x, v_y, marker='o', lw=3)

    ax2.set_title("Model Dice Coefficient", size=20)
    ax2.set_ylabel("Dice Coefficient", size=20)
    ax2.set_xlabel("Epoch", size=20)

    ax2.legend(["Training", "Validation"], loc="upper left")

    ax2.set_xticks(x)
    ax2.set_xlim(0, )
    ax2.set_ylim(0, )

    ax2.axvline(best_epoch, lw=3, color="red")

    ax2.tick_params(axis="y", labelsize=16)
    ax2.tick_params(axis="x", labelsize=12, rotation=90)

    fig.tight_layout()

    fig.savefig(
        replace_extension(
            remove_tags(file_location, skip_tags),
            suffix + ext
        ),
        bbox_inches="tight"
    )


def export_loss_mse_coeff(history, tf, TCL, RR, ARC, file_location, suffix="_model_loss_mse_coeff", ext=".png",
                          style="ggplot", log_base=10, skip_tags="_{epoch}"):
    plt.style.use(style)
    fig, ([ax1, ax2, ax3, ax4], [ax5, ax6, ax7, ax8]) = plt.subplots(2, 4, sharex=False, sharey=False, figsize=(24, 12))
    fig.suptitle(
        'Training and Validation: Loss, Mean Squared Error and Coefficiennt of Determination for PCPC training on ' + TCL + '\n' +
        ' with random ratio set at ' + str(RR) + '  for ' + tf + ' ' + ARC + ' architecture'  '\n \n', fontsize=24)

    t_y = history.history['loss']
    t_x = [int(i) for i in range(1, len(t_y) + 1)]

    v_y = history.history["val_loss"]
    v_x = [int(i) for i in range(1, len(v_y) + 1)]

    ax1.plot(t_x, t_y, marker='o')
    ax1.plot(v_x, v_y, marker='o')

    ax1.set_xticks(t_x)

    ax1.set_title("Model loss")
    ax1.set_ylabel("Loss")
    ax1.set_xlabel("Epoch")
    ax1.legend(["Training", "Validation"], loc="upper right")

    t_y = history.history['loss']
    t_x = [int(i) for i in range(1, len(t_y) + 1)]

    v_y = history.history["val_loss"]
    v_x = [int(i) for i in range(1, len(v_y) + 1)]

    ax2.plot(t_x, t_y, marker='o')
    ax2.plot(v_x, v_y, marker='o')

    ax2.set_xticks(t_x)
    ax2.set_yscale('log')

    ax2.set_title("Model Log scale loss")
    ax2.set_ylabel("Loss Log-Scale")
    ax2.set_xlabel("Epoch")
    ax2.legend(["Training", "Validation"], loc="upper right")

    t_y = history.history['coeff_determination']
    t_x = [int(i) for i in range(1, len(t_y) + 1)]

    v_y = history.history["val_coeff_determination"]
    v_x = [int(i) for i in range(1, len(v_y) + 1)]

    ax3.plot(t_x, t_y, marker='o')
    ax3.plot(v_x, v_y, marker='o')

    ax3.set_xticks(t_x)
    # ax3.set_ylim([0, 1])

    ax3.set_title("R Squared")
    ax3.set_ylabel("R Squared")
    ax3.set_xlabel("Epoch")
    ax3.legend(["Training", "Validation"], loc="upper left")

    t_y = history.history['coeff_determination']
    t_x = [int(i) for i in range(1, len(t_y) + 1)]

    v_y = history.history["val_coeff_determination"]
    v_x = [int(i) for i in range(1, len(v_y) + 1)]

    ax4.plot(t_x, t_y, marker='o')
    ax4.plot(v_x, v_y, marker='o')

    ax4.set_xticks(t_x)
    ax4.set_ylim([0, 1])

    ax4.set_title("R Squared")
    ax4.set_ylabel("R Squared")
    ax4.set_xlabel("Epoch")
    ax4.legend(["Training", "Validation"], loc="upper left")

    t_y = history.history['precision']
    t_x = [int(i) for i in range(1, len(t_y) + 1)]

    v_y = history.history["val_precision"]
    v_x = [int(i) for i in range(1, len(v_y) + 1)]

    ax5.plot(t_x, t_y, marker='o')
    ax5.plot(v_x, v_y, marker='o')
    ax5.set_xticks(t_x)
    ax5.set_ylim([0, 1])

    ax5.set_title("Precision")
    ax5.set_ylabel("Precision")
    ax5.set_xlabel("Epoch")
    ax5.legend(["Training", "Validation"], loc="upper left")

    #
    t_y = history.history['recall']
    t_x = [int(i) for i in range(1, len(t_y) + 1)]

    v_y = history.history["val_recall"]
    v_x = [int(i) for i in range(1, len(v_y) + 1)]

    ax6.plot(t_x, t_y, marker='o')
    ax6.plot(v_x, v_y, marker='o')

    ax6.set_xticks(t_x)
    ax6.set_ylim([0, 1])

    ax6.set_title("Recall")
    ax6.set_ylabel("Recall")
    ax6.set_xlabel("Epoch")
    ax6.legend(["Training", "Validation"], loc="upper left")
    #
    t_y = history.history['pearson']
    t_x = [int(i) for i in range(1, len(t_y) + 1)]

    v_y = history.history["val_pearson"]
    v_x = [int(i) for i in range(1, len(v_y) + 1)]

    ax7.plot(t_x, t_y, marker='o')
    ax7.plot(v_x, v_y, marker='o')

    ax7.set_xticks(t_x)
    ax7.set_ylim([0, 1])

    ax7.set_title("Pearson Correlation")
    ax7.set_ylabel("Pearson Correlation")
    ax7.set_xlabel("Epoch")
    ax7.legend(["Training", "Validation"], loc="upper left")
    #
    t_y = history.history['spearman']
    t_x = [int(i) for i in range(1, len(t_y) + 1)]

    v_y = history.history["val_spearman"]
    v_x = [int(i) for i in range(1, len(v_y) + 1)]

    ax8.plot(t_x, t_y, marker='o')
    ax8.plot(v_x, v_y, marker='o')

    ax8.set_xticks(t_x)
    ax8.set_ylim([0, 1])

    ax8.set_title("Spearman Correlation")
    ax8.set_ylabel("Spearman Correlation")
    ax8.set_xlabel("Epoch")
    ax8.legend(["Training", "Validation"], loc="upper left")

    fig.tight_layout(pad=10)

    fig.savefig(
        replace_extension(
            remove_tags(file_location, skip_tags),
            suffix + ext
        ),
        bbox_inches="tight"
    )


def export_prc(precision, recall, file_location, title="Precision Recall Curve", suffix="_prc", ext=".png",
               style="ggplot"):
    plt.style.use(style)

    plt.plot(recall, precision)

    plt.title(title)
    plt.ylabel("Precision")
    plt.xlabel("Recall")

    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])

    plt.savefig(
        replace_extension(file_location, suffix + ext),
        bbox_inches="tight"
    )

    plt.close("all")


def plot_chromosome_scores_dist(input_bigwig, chrom_name, region_start, region_stop):
    with pyBigWig.open(input_bigwig) as input_bw:
        chr_vals = input_bw.values(chrom_name, region_start, region_stop, numpy=True)

    plt.hist(chr_vals[chr_vals > 0])
