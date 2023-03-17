# This is the file for interpretation of transformer models
import tensorflow as tf
import numpy as np
from shap import DeepExplainer, GradientExplainer

from maxatac.utilities.transformer_interpret_tools import dinuc_shuffle, get_data, load_model_from_dir

def run_transformer_interpret(args):
    """
    The main place for all transformer interpretation functions
    """
    if args.analysis == "":
        return
    
    if args.analysis == "shap":
        attribution_shap(
            args.meta_file, args.chromosome, args.cell_type, args.output_dir, args.model_base_dir
        )


def attribution_shap(meta_file, chromosome, cell_type, output_dir, model_base_dir, tasks=15, num_background_seqs=1000):
    """
    Implement Integrated Gradients for the model
    """
    #tf.compat.v1.disable_v2_behavior()
    #inputs, targets = get_data(meta_file, chromosome, cell_type, output_dir)     # numpy array shape (batch, 1024, 5) (let's set batch = 1)
    
    with open("/data/weirauchlab/team/ngun7t/maxatac/scratch/input.npy", "rb") as f:
        inputs = np.load(f)
    with open("/data/weirauchlab/team/ngun7t/maxatac/scratch/target.npy", "rb") as f:
        targets = np.load(f)
    
    # Load model from dir
    model = load_model_from_dir(model_base_dir)

    # Extract genome sequence and ATAC seq
    one_hot_encoded_genome = inputs[0, :, :4]
    atac_signal = np.expand_dims(inputs[0:1, :, 1], axis=-1)
    
    # get background data
    background = dinuc_shuffle(one_hot_encoded_genome, num_background_seqs)
    background_with_atac = np.concatenate([background, np.repeat(atac_signal, repeats=num_background_seqs, axis=0)], axis=-1)
    
    # probably run deepshap
    explainer = GradientExplainer(
        model=([model.input], model.layers[-1].output),
        data=background_with_atac
    )

    contribs = explainer.shap_values(inputs[0:10, :, :])
    with open("/data/weirauchlab/team/ngun7t/maxatac/scratch/shap.npy", "wb") as f:
        np.save(f, contribs)


def conv_tower_output(meta_file, chromosome, cell_type, output_dir):
    """
    Get the output of the last conv tower layer to see how well this vector representation has already achieved in the classification
    Some methods include PCA - tSNE - UMAP of these representations
    Or building probing classifiers (https://arxiv.org/pdf/2102.12452.pdf)
    """
    model_base_dir = None
    model = load_model_from_dir(model_base_dir)
    last_conv_tower_layer = "downsampling_conv"
    num_seqs = 10

    # Get the output of the last conv layer (with the name downsampling_conv) and create a sub model
    last_conv = [layer.name for layer in model.layers if last_conv_tower_layer in layer.name][-1]
    sub_model = tf.keras.Model(model.input, model.get_layer(last_conv).output)

    # Get a slice of the data
    #a, b = get_data(meta_file, chromosome, cell_type, output_dir)
    # For debugging, load the data instead
    with open("/data/weirauchlab/team/ngun7t/maxatac/scratch/input.npy", "rb") as f:
        inputs = np.load(f)
    with open("/data/weirauchlab/team/ngun7t/maxatac/scratch/target.npy", "rb") as f:
        targets = np.load(f)

    # Choose inputs that have high target peaks
    num_peaks, num_peak_counts = map(list, np.unique(targets, return_counts=True))


    # Make a forward pass through the model and get the output 




def check_attention_correlation():
    """
    Run the experiments for checking the attention weights across the best performing models
    and across the best epochs within the same model
    """
    # 