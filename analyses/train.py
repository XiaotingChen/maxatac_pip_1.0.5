import logging
import sys
import os
import numpy as np
import json
import shutil
import timeit
import tensorflow

from keras.utils.data_utils import OrderedEnqueuer

from maxatac.utilities.constants import TRAIN_MONITOR, NUM_HEADS, NUM_MHA
from maxatac.utilities.system_tools import Mute
from maxatac.utilities.phuc_utilities import generate_numpy_arrays

with Mute():
    from tensorflow.keras.models import load_model
    from maxatac.utilities.callbacks import get_callbacks
    from maxatac.utilities.training_tools import DataGenerator, MaxATACModel, ROIPool, SeqDataGenerator, model_selection, save_metadata
    from maxatac.utilities.plot import export_binary_metrics, export_loss_mse_coeff, export_model_structure, plot_attention_weights


def run_training(args):
    """
    Train a maxATAC model using ATAC-seq and ChIP-seq data

    The primary input to the training function is a meta file that contains all of the information for the locations of
    ATAC-seq signal, ChIP-seq signal, TF, and Cell type.

    Example header for meta file. The meta file must be a tsv file, but the order of the columns does not matter. As
    long as the column names are the same:

    TF | Cell_Type | ATAC_Signal_File | Binding_File | ATAC_Peaks | ChIP_peaks

    ## An example meta file is included in our repo

    _________________
    Workflow Overview

    1) Set up the directories and filenames
    2) Initialize the model based on the desired architectures
    3) Read in training and validation pools
    4) Initialize the training and validation generators
    5) Fit the models with the specific parameters

    :params args: arch, seed, output, prefix, output_activation, lrate, decay, weights,
    dense, batch_size, val_batch_size, train roi, validate roi, meta_file, sequence, average, threads, epochs, batches,
    tchroms, vchroms, shuffle_cell_type, rev_comp

    :returns: Trained models saved after each epoch
    """
    # Check if tf is using GPU
    assert len(tensorflow.config.list_physical_devices('GPU')) > 0, "Tensorflow is not using GPU"
    gpus = tensorflow.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tensorflow.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tensorflow.config.list_logical_devices('GPU')
            logging.error(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    
    # Start Timer
    startTime = timeit.default_timer()

    # Save metadata
    save_metadata(args.output, args)

    logging.error("Set up model parameters")

    
    # Initialize the model with the architecture of choice
    maxatac_model = MaxATACModel(arch=args.arch,
                                 seed=args.seed,
                                 output_directory=args.output,
                                 prefix=args.prefix,
                                 threads=args.threads,
                                 meta_path=args.meta_file,
                                 output_activation=args.output_activation,
                                 dense=args.dense,
                                 weights=args.weights
                                 )

    ## Save the args and the constants to the output folder
    #with open(os.path.join(args.output, "user_args.txt"), "w") as f:
    #    json.dump(args.__dict__, f, indent=2)

    # This is only for this project, so please change this line of code
    shutil.copyfile(
        "/users/ngun7t/anaconda3/envs/maxatac/lib/python3.9/site-packages/maxatac/utilities/constants.py",
        os.path.join(args.output, "constants.py")
    )
    
    logging.error("Import training regions")

    # The args.train_roi and the args.validate_roi are the BED files that specify the regions of interest on the genome
    # Import training regions
    train_examples = ROIPool(chroms=args.tchroms,
                             roi_file_path=args.train_roi,
                             meta_file=args.meta_file,
                             prefix=args.prefix,
                             output_directory=maxatac_model.output_directory,
                             shuffle=True,
                             tag="training")

    # Import validation regions
    validate_examples = ROIPool(chroms=args.vchroms,
                                roi_file_path=args.validate_roi,
                                meta_file=args.meta_file,
                                prefix=args.prefix,
                                output_directory=maxatac_model.output_directory,
                                shuffle=True,
                                tag="validation")

    logging.error("Initialize data generator")

    # The function returns a generator that each next() yields an input_batch and a target_batch, which in ML language might be features X and labels y
    # There's the rand_ratio that defines the ratio at which the random genome regions and the ROI regions are taken
    # Initialize the training generator
    train_gen = DataGenerator(sequence=args.sequence,
                              meta_table=maxatac_model.meta_dataframe,
                              roi_pool=train_examples.ROI_pool,
                              cell_type_list=maxatac_model.cell_types,
                              rand_ratio=args.rand_ratio,
                              chroms=args.tchroms,
                              batch_size=args.batch_size,
                              shuffle_cell_type=args.shuffle_cell_type,
                              rev_comp_train=args.rev_comp
                              )

    # (beta) load a batch of images using train_gen, take one sample as the data sample
    input_batch, _ = next(iter(train_gen))

    # Create keras.utils.sequence object from training generator
    seq_train_gen = SeqDataGenerator(batches=args.batches, generator=train_gen)

    # Builds a Enqueuer from a Sequence.
    # I think the point of using Enqueuer is to use parallelism, but I don't know then why use_multiprocessing is set to False
    '''train_gen_enq = OrderedEnqueuer(seq_train_gen, use_multiprocessing=True)
    train_gen_enq.start(workers=args.threads, max_queue_size=args.threads * 2)'''
    train_gen_enq = OrderedEnqueuer(seq_train_gen, use_multiprocessing=False)
    train_gen_enq.start(workers=1, max_queue_size=args.threads * 2)
    enq_train_gen = train_gen_enq.get()     # enq_train_gen is now a generator to extract data from the queue

    # Initialize the validation generator
    val_gen = DataGenerator(sequence=args.sequence,
                            meta_table=maxatac_model.meta_dataframe,
                            roi_pool=validate_examples.ROI_pool,
                            cell_type_list=maxatac_model.cell_types,
                            rand_ratio=args.rand_ratio,
                            chroms=args.vchroms,
                            batch_size=args.batch_size,
                            shuffle_cell_type=args.shuffle_cell_type,
                            rev_comp_train=args.rev_comp
                            )

    # Create keras.utils.sequence object from validation generator
    seq_validate_gen = SeqDataGenerator(batches=args.batches, generator=val_gen)

    # Builds a Enqueuer from a Sequence.
    '''val_gen_enq = OrderedEnqueuer(seq_validate_gen, use_multiprocessing=True)
    val_gen_enq.start(workers=args.threads, max_queue_size=args.threads * 2)'''
    val_gen_enq = OrderedEnqueuer(seq_validate_gen, use_multiprocessing=False)
    val_gen_enq.start(workers=1, max_queue_size=args.threads * 2)
    enq_val_gen = val_gen_enq.get()


    # Fit the model
    logging.error("Start training the model")
    training_history = maxatac_model.nn_model.fit(enq_train_gen,    # model.fit() accepts a generator or Sequence that returns (inputs, targets). From the doc, when x is a generator, y should not be specified
                                                validation_data=enq_val_gen,
                                                steps_per_epoch=args.batches,
                                                validation_steps=args.batches,
                                                epochs=args.epochs,
                                                callbacks=get_callbacks(
                                                    model_location=maxatac_model.results_location,
                                                    log_location=maxatac_model.log_location,
                                                    tensor_board_log_dir=maxatac_model.tensor_board_log_dir,
                                                    monitor=TRAIN_MONITOR
                                                    ),
                                                max_queue_size=10,
                                                use_multiprocessing=False,
                                                workers=1,
                                                verbose=1
                                                )

    logging.error("Plot and save results")

    # Select best model
    best_epoch = model_selection(training_history=training_history,
                                 output_dir=maxatac_model.output_directory)

    # If plot then plot the model structure and training metrics
    # If model is transformer then also plot all the attention weights and final positional encoding
    if args.plot:
        tf = maxatac_model.train_tf
        TCL = '_'.join(maxatac_model.cell_types)
        ARC = args.arch
        RR = args.rand_ratio

        export_binary_metrics(training_history, tf, RR, ARC, maxatac_model.results_location, best_epoch)
        export_model_structure(maxatac_model.nn_model, maxatac_model.results_location)

        data_sample = tensorflow.expand_dims(input_batch[0], axis=0)
        mha_names = [f"Encoder_{i}_softmax_att_weights" for i in range(NUM_MHA)]
        plot_attention_weights(maxatac_model.nn_model, mha_names, data_sample, num_heads=NUM_HEADS, file_location=args.output)

    logging.error("Results are saved to: " + maxatac_model.results_location)
    
    # Measure End Time of Training
    stopTime = timeit.default_timer()
    totalTime = stopTime - startTime
    
    # Output running time in a nice format.
    mins, secs = divmod(totalTime, 60)
    hours, mins = divmod(mins, 60)

    logging.error("Total training time: %d:%d:%d.\n" % (hours, mins, secs))

    sys.exit()
    