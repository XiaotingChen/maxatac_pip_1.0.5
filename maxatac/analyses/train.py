import logging
import sys
import os
import numpy as np
import json
import shutil
import timeit

import pandas as pd
import tensorflow

# from keras.utils.data_utils import OrderedEnqueuer
from tensorflow.keras.utils import OrderedEnqueuer

from maxatac.utilities.constants import (
    TRAIN_MONITOR,
    INPUT_LENGTH,
    INPUT_CHANNELS,
    OUTPUT_LENGTH,
)
from maxatac.utilities.system_tools import Mute
from maxatac.utilities.phuc_utilities import generate_numpy_arrays

with Mute():
    from tensorflow.keras.models import load_model
    from maxatac.utilities.callbacks import get_callbacks
    from maxatac.utilities.training_tools import (
        DataGenerator,
        DataGenerator_v2,
        MaxATACModel,
        ROIPool,
        SeqDataGenerator,
        model_selection,
        save_metadata,
        CHIP_sample_weight_adjustment,
        ValidDataGen,
        DataGen,
        peak_centric_map,
        random_shuffling_map
    )
    from maxatac.utilities.plot import (
        export_binary_metrics,
        export_loss_mse_coeff,
        export_model_structure,
        plot_attention_weights,
    )
    from maxatac.utilities.genome_tools import (
        build_chrom_sizes_dict,
    )


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
    logging.error(args)

    gpus = tensorflow.config.list_physical_devices("GPU")
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tensorflow.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tensorflow.config.list_logical_devices("GPU")
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    if tensorflow.test.gpu_device_name():
        print("GPU device found")
    else:
        print("No GPU found")

    # Start Timer
    startTime = timeit.default_timer()

    logging.error("Set up model parameters")

    # Read model config
    with open(args.model_config, "r") as f:
        model_config = json.load(f)

    # override model config from maxatac args
    model_config["OPTIMIZER"] = args.optimizer

    model_config["USING_BASENJI_KERNEL"] = args.USING_BASENJI_KERNEL
    model_config["USING_ENFORMER_KERNEL"] = args.USING_ENFORMER_KERNEL
    model_config["BASENJI_KERNEL_TRAINABLE"] = args.BASENJI_KERNEL_TRAINABLE
    model_config["ENFORMER_KERNEL_TRAINABLE"] = args.ENFORMER_KERNEL_TRAINABLE
    model_config["KERNEL_REPLACING"] = args.KERNEL_REPLACING

    model_config["SUPPRESS_DROPOUT"] = args.SUPPRESS_DROPOUT
    model_config[
        "RESIDUAL_CONNECTION_DROPOUT_RATE"
    ] = args.RESIDUAL_CONNECTION_DROPOUT_RATE
    model_config["PREDICTION_HEAD_DROPOUT_RATE"] = args.PREDICTION_HEAD_DROPOUT_RATE

    model_config["COSINEDECAYRESTARTS"] = args.COSINEDECAYRESTARTS
    model_config[
        "COSINEDECAYRESTARTS_FIRST_DECAY_STEPS"
    ] = args.COSINEDECAYRESTARTS_FIRST_DECAY_STEPS

    model_config["FOCAL_LOSS_ALPHA"] = args.FOCAL_LOSS_ALPHA
    model_config["FOCAL_LOSS_GAMMA"] = args.FOCAL_LOSS_GAMMA
    model_config["FOCAL_LOSS"] = args.FOCAL_LOSS
    model_config["FOCAL_LOSS_APPLY_ALPHA"] = args.FOCAL_LOSS_APPLY_ALPHA

    model_config["INITIAL_LEARNING_RATE"] = args.INITIAL_LEARNING_RATE
    model_config["REGULARIZATION"] = args.REGULARIZATION
    model_config["ELASTIC_L1"] = args.ELASTIC_L1
    model_config["ELASTIC_L2"] = args.ELASTIC_L2

    # Initialize the model with the architecture of choice
    maxatac_model = MaxATACModel(
        arch=args.arch,
        seed=args.seed,
        model_config=model_config,
        output_directory=args.output,
        prefix=args.prefix,
        threads=args.threads,
        meta_path=args.meta_file,
        output_activation=args.output_activation,
        dense=args.dense,
        weights=args.weights,
        inter_fusion=model_config["INTER_FUSION"],
    )

    # export model structure
    export_model_structure(
        maxatac_model.nn_model, maxatac_model.results_location, ext=".pdf"
    )

    ## Save the args and the constants to the output folder
    # with open(os.path.join(args.output, "user_args.txt"), "w") as f:
    #    json.dump(args.__dict__, f, indent=2)

    logging.error("Import training regions")

    # The args.train_roi and the args.validate_roi are the BED files that specify the regions of interest on the genome

    # Import training regions
    train_examples = ROIPool(
        chroms=args.tchroms,
        roi_file_path=args.train_roi,
        meta_file=args.meta_file,
        prefix=args.prefix,
        output_directory=maxatac_model.output_directory,
        shuffle=True,
        tag="training",
    )

    # Import validation regions
    validate_examples = ROIPool(
        chroms=args.vchroms,
        roi_file_path=args.validate_roi,
        meta_file=args.meta_file,
        prefix=args.prefix,
        output_directory=maxatac_model.output_directory,
        shuffle=True,
        tag="validation",
    )

    if args.ATAC_Sampling_Multiplier != 0:
        steps_per_epoch_v2 = int(
            train_examples.ROI_pool_CHIP.shape[0]
            // np.ceil((args.batch_size / (1.0 + float(args.ATAC_Sampling_Multiplier))))
        )
        validation_steps_v2 = int(
            validate_examples.ROI_pool.shape[0] // args.batch_size
        )  # under a fixed ratio of 5, we are probably just under-sample to 1/3 of the total background

        # Save metadata
        save_metadata(
            args.output,
            args,
            model_config,
            extra={
                "training CHIP ROI total regions": train_examples.ROI_pool_CHIP.shape[
                    0
                ],
                "training ATAC ROI total regions": train_examples.ROI_pool_ATAC.shape[
                    0
                ],
                "validate CHIP ROI total regions": validate_examples.ROI_pool_CHIP.shape[
                    0
                ],
                "validate ATAC ROI total regions": validate_examples.ROI_pool_ATAC.shape[
                    0
                ],
                "training CHIP ROI unique regions": train_examples.ROI_pool_unique_region_size_CHIP,
                "training ATAC ROI unique regions": train_examples.ROI_pool_unique_region_size_ATAC,
                "validate CHIP ROI unique regions": validate_examples.ROI_pool_unique_region_size_CHIP,
                "validate ATAC ROI unique regions": validate_examples.ROI_pool_unique_region_size_ATAC,
                "batch size": args.batch_size,
                "training batches per epoch": steps_per_epoch_v2,
                "validation batches per epoch": validation_steps_v2,
                "total epochs": args.epochs,
                "ATAC_Sampling_Multiplier": args.ATAC_Sampling_Multiplier,
                "CHIP_Sample_Weight_Baseline": args.CHIP_Sample_Weight_Baseline,
            },
        )

        # annotate CHIP ROI with additional sample weight adjustment
        train_examples.ROI_pool_CHIP = CHIP_sample_weight_adjustment(
            train_examples.ROI_pool_CHIP
        )
        validate_examples.ROI_pool_CHIP = CHIP_sample_weight_adjustment(
            validate_examples.ROI_pool_CHIP
        )

    logging.error("Initialize data generator")

    if args.get_tfds:
        data_meta = pd.DataFrame(
            columns=["train or valid", "tf", "cell_type", "roi_type", "path"]
        )
        chr_limit = build_chrom_sizes_dict(
            chrom_sizes_filename=args.chromosome_size_file
        )
        tf = args.meta_file.split("/")[-1].split(".")[0].split("meta_file_")[1]

        print("Getting valid samples")
        # valid data, only with extended input size
        data = tensorflow.data.Dataset.from_generator(
            ValidDataGen(
                sequence=args.sequence,
                meta_table=maxatac_model.meta_dataframe,
                roi_pool_atac=validate_examples.ROI_pool_ATAC,
                roi_pool_chip=validate_examples.ROI_pool_CHIP,
                cell_type_list=maxatac_model.cell_types,
                atac_sampling_multiplier=args.ATAC_Sampling_Multiplier,
                chip_sample_weight_baseline=args.CHIP_Sample_Weight_Baseline,
                batch_size=args.batch_size,
                shuffle=True,
                chr_limit=chr_limit,
                flanking_padding_size=512,
            ),
            output_signature=(
                tensorflow.TensorSpec(
                    shape=(INPUT_LENGTH + 2 * args.flanking_size, INPUT_CHANNELS),
                    dtype=tensorflow.float32,
                ),
                tensorflow.TensorSpec(
                    shape=(OUTPUT_LENGTH * 2), dtype=tensorflow.float32
                ),
                tensorflow.TensorSpec(shape=(), dtype=tensorflow.float32),
            ),
        )
        data_path = "{}/{}/{}".format(
            args.tfds_path,"valid", tf
        )
        data.save(
            path=data_path,
            compression="GZIP",
        )
        data_meta.loc[data_meta.shape[0]] = ["valid", tf, '.', '.', data_path]

        print("Getting train samples")

        # atac # todo: need to parallize this block into smaller chunks
        print("ATAC")
        data = tensorflow.data.Dataset.from_generator(
            DataGen(
                sequence=args.sequence,
                meta_table=maxatac_model.meta_dataframe,
                roi_pool=train_examples.ROI_pool_ATAC,
                chip=False,
                cell_type=None,
                atac_sampling_multiplier=args.ATAC_Sampling_Multiplier,
                chip_sample_weight_baseline=args.CHIP_Sample_Weight_Baseline,
                batch_size=args.batch_size,
                shuffle=True,
                chr_limit=chr_limit,
                flanking_padding_size=args.flanking_size,
            ),
            output_signature=(
                tensorflow.TensorSpec(
                    shape=(INPUT_LENGTH + 2 * args.flanking_size, INPUT_CHANNELS),
                    dtype=tensorflow.float32,
                ),
                tensorflow.TensorSpec(
                    shape=(OUTPUT_LENGTH * 2), dtype=tensorflow.float32
                ),
                tensorflow.TensorSpec(shape=(), dtype=tensorflow.float32),
            ),
        )
        data_path = (
            "{}/{}/{}_{}_{}".format(
                args.tfds_path, "train", tf, '.', "ATAC"
            )
        )
        data.save(
            path=data_path,
            compression="GZIP",
        )
        data_meta.loc[data_meta.shape[0]] = ["train", tf, '.', 'ATAC', data_path]

        # training dataset, augmented by cell type shuffling, and extended input size
        for cell_type in maxatac_model.cell_types:
            # chip
            print("CHIP",cell_type,sep="\t")
            data = tensorflow.data.Dataset.from_generator(
                DataGen(
                    sequence=args.sequence,
                    meta_table=maxatac_model.meta_dataframe,
                    roi_pool=train_examples.ROI_pool_CHIP,
                    chip=True,
                    cell_type=cell_type,
                    atac_sampling_multiplier=args.ATAC_Sampling_Multiplier,
                    chip_sample_weight_baseline=args.CHIP_Sample_Weight_Baseline,
                    batch_size=args.batch_size,
                    shuffle=True,
                    chr_limit=chr_limit,
                    flanking_padding_size=512,
                ),
                output_signature=(
                    tensorflow.TensorSpec(
                        shape=(INPUT_LENGTH + 2 * args.flanking_size, INPUT_CHANNELS),
                        dtype=tensorflow.float32,
                    ),
                    tensorflow.TensorSpec(
                        shape=(OUTPUT_LENGTH * 2), dtype=tensorflow.float32
                    ),
                    tensorflow.TensorSpec(shape=(), dtype=tensorflow.float32),
                ),
            )
            data_path = (
                "{}/{}/{}_{}_{}".format(
                    args.tfds_path,"train", tf, cell_type, "CHIP"
                )
            )
            data.save(
                path=data_path,
                compression="GZIP",
            )
            data_meta.loc[data_meta.shape[0]] = ["train", tf, cell_type, 'CHIP', data_path]

        data_meta.to_csv(args.tfds_meta, header=True, index=False, sep="\t")
        logging.error("Generate tfds completed!")
        sys.exit()

    # # Initialize the training generator
    # if args.ATAC_Sampling_Multiplier == 0:
    #     train_gen = DataGenerator(
    #         sequence=args.sequence,
    #         meta_table=maxatac_model.meta_dataframe,
    #         roi_pool=train_examples.ROI_pool,
    #         cell_type_list=maxatac_model.cell_types,
    #         rand_ratio=args.rand_ratio,
    #         chroms=args.tchroms,
    #         batch_size=args.batch_size,
    #         shuffle_cell_type=args.shuffle_cell_type,
    #         rev_comp_train=args.rev_comp,
    #         inter_fusion=model_config["INTER_FUSION"],
    #     )
    # else:
    #     train_gen = DataGenerator_v2(
    #         sequence=args.sequence,
    #         meta_table=maxatac_model.meta_dataframe,
    #         roi_pool_atac=train_examples.ROI_pool_ATAC,
    #         roi_pool_chip=train_examples.ROI_pool_CHIP,
    #         cell_type_list=maxatac_model.cell_types,
    #         rand_ratio=args.rand_ratio,
    #         chroms=args.tchroms,
    #         batch_size=args.batch_size,
    #         shuffle_cell_type=args.shuffle_cell_type,
    #         rev_comp_train=args.rev_comp,
    #         inter_fusion=False,
    #         atac_sampling_multiplier=args.ATAC_Sampling_Multiplier,
    #         chip_sample_weight_baseline=args.CHIP_Sample_Weight_Baseline,
    #     )
    #
    # # Create keras.utils.sequence object from training generator
    # seq_train_gen = SeqDataGenerator(batches=args.batch_size, generator=train_gen)

    # Specify max_que_size
    if args.max_queue_size:
        queue_size = int(args.max_queue_size)
        logging.info("User specified Max Queue Size: " + str(queue_size))
    else:
        queue_size = args.threads * 2
        logging.info("Max Queue Size found: " + str(queue_size))

    # # Builds a Enqueuer from a Sequence.
    # # Specify multiprocessing
    # if args.multiprocessing:
    #     logging.info("Training with multiprocessing")
    #     train_gen_enq = OrderedEnqueuer(seq_train_gen, use_multiprocessing=True)
    #     train_gen_enq.start(workers=args.threads, max_queue_size=queue_size)
    #
    # else:
    #     logging.info("Training without multiprocessing")
    #     train_gen_enq = OrderedEnqueuer(seq_train_gen, use_multiprocessing=False)
    #     train_gen_enq.start(workers=1, max_queue_size=queue_size)
    #
    # enq_train_gen = (
    #     train_gen_enq.get()
    # )  # enq_train_gen is now a generator to extract data from the queue
    #
    # # validation tfds
    # valid_data = tensorflow.data.Dataset.from_generator(
    #     ValidDataGen(
    #         sequence=args.sequence,
    #         meta_table=maxatac_model.meta_dataframe,
    #         roi_pool_atac=validate_examples.ROI_pool_ATAC,
    #         roi_pool_chip=validate_examples.ROI_pool_CHIP,
    #         cell_type_list=maxatac_model.cell_types,
    #         atac_sampling_multiplier=args.ATAC_Sampling_Multiplier,
    #         chip_sample_weight_baseline=args.CHIP_Sample_Weight_Baseline,
    #         batch_size=args.batch_size,
    #         shuffle=True,
    #     ),
    #     output_signature=(
    #         tensorflow.TensorSpec(shape=(1024, 5), dtype=tensorflow.float32),
    #         tensorflow.TensorSpec(shape=(32), dtype=tensorflow.float32),
    #         tensorflow.TensorSpec(shape=(), dtype=tensorflow.float32),
    #     ),
    # )

    # # Initialize the validation generator
    # if args.ATAC_Sampling_Multiplier == 0:
    #     val_gen = DataGenerator(
    #         sequence=args.sequence,
    #         meta_table=maxatac_model.meta_dataframe,
    #         roi_pool=validate_examples.ROI_pool,
    #         cell_type_list=maxatac_model.cell_types,
    #         rand_ratio=args.rand_ratio,
    #         chroms=args.vchroms,
    #         batch_size=args.batch_size,
    #         shuffle_cell_type=args.shuffle_cell_type,
    #         rev_comp_train=args.rev_comp,
    #         inter_fusion=model_config["INTER_FUSION"],
    #     )
    # else:
    #     val_gen = DataGenerator_v2(
    #         sequence=args.sequence,
    #         meta_table=maxatac_model.meta_dataframe,
    #         roi_pool_atac=validate_examples.ROI_pool_ATAC,
    #         roi_pool_chip=validate_examples.ROI_pool_CHIP,
    #         cell_type_list=maxatac_model.cell_types,
    #         rand_ratio=args.rand_ratio,
    #         chroms=args.vchroms,
    #         batch_size=args.batch_size,
    #         shuffle_cell_type=False,
    #         rev_comp_train=args.rev_comp,
    #         inter_fusion=model_config["INTER_FUSION"],
    #         atac_sampling_multiplier=args.ATAC_Sampling_Multiplier,
    #         chip_sample_weight_baseline=args.CHIP_Sample_Weight_Baseline,
    #     )
    #
    # # Create keras.utils.sequence object from validation generator
    # seq_validate_gen = SeqDataGenerator(batches=args.batch_size, generator=val_gen)
    #
    # # Builds a Enqueuer from a Sequence.
    # # Specify multiprocessing
    # if args.multiprocessing:
    #     logging.info("Validating with multiprocessing")
    #     val_gen_enq = OrderedEnqueuer(seq_validate_gen, use_multiprocessing=True)
    #     val_gen_enq.start(workers=args.threads, max_queue_size=queue_size)
    # else:
    #     logging.info("Validating without multiprocessing")
    #     val_gen_enq = OrderedEnqueuer(seq_validate_gen, use_multiprocessing=False)
    #     val_gen_enq.start(workers=1, max_queue_size=queue_size)
    # enq_val_gen = val_gen_enq.get()


    # get tfds train and valid object

    data_meta=pd.read_csv(args.tfds_meta,header=0,sep='\t')

    # valid data
    valid_tfds_file = data_meta[data_meta['train or valid'] == 'valid']['path'].values[0]
    valid_tfds=tensorflow.data.Dataset.load(valid_tfds_file,compression="GZIP",)
    valid_data=(valid_tfds
                .take(
                    (validate_examples.ROI_pool.shape[0] // args.batch_size)
                    * args.batch_size
                )
                .map(peak_centric_map)
                .cache()
                .repeat(args.epochs)
                .batch(
                    batch_size=args.batch_size,
                    num_parallel_calls=tensorflow.data.AUTOTUNE,
                    drop_remainder=True,
                    deterministic=False,
                )
                .prefetch(tensorflow.data.AUTOTUNE)
    )

    # train data
    # atac
    atac_tfds_file = data_meta[(data_meta['train or valid'] == 'train') & (data_meta['roi_type'] == 'ATAC')]['path'].values[0]
    atac_tfds=tensorflow.data.Dataset.load(atac_tfds_file,compression="GZIP",)

    # chip
    chip_tfds = []
    for cell_type, file_path in data_meta[(data_meta['train or valid'] == 'train') & (data_meta['roi_type'] == 'CHIP')][['cell_type', 'path']].values:
        if maxatac_model.meta_dataframe[maxatac_model.meta_dataframe["Cell_Line"] == cell_type]['Train_Test_Label'] == "Train":
            tfds=tensorflow.data.Dataset.load(file_path, compression="GZIP", )
            data=tfds.map(peak_centric_map).cache().shuffle(tfda.cardinality().numpy()).repeat(args.epochs)
            chip_tfds.append(data)


    _chip_size=len(chip_tfds)
    _chip_prob=1.0/(1.0+float(args.ATAC_Sampling_Multiplier))
    _atac_prob=(1.0-_chip_prob)

    train_data_chip=tf.data.Dataset.sample_from_datasets(
        chip_tfds,
        weights=[1.0/float(_chip_size)]*_chip_size,
        stop_on_empty_dataset=False,
        rerandomize_each_iteration=True
    )

    train_data = (tf.data.Dataset.sample_from_datasets(
        [train_data_chip,atac_tfds.map(peak_centric_map).cache().repeat(args.epochs)],
        weights=[_chip_prob,_atac_prob],
        stop_on_empty_dataset=False,
        rerandomize_each_iteration=True
        )
        .batch(batch_size= args.batch_size,
            num_parallel_calls=tensorflow.data.AUTOTUNE,
            drop_remainder=True,
            deterministic=False,
        )
        .prefetch(tensorflow.data.AUTOTUNE)
    )

    # Fit the model
    training_history = maxatac_model.nn_model.fit(
        train_data,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch_v2,
        validation_data=valid_data,
        validation_steps=validation_steps_v2,
        callbacks=get_callbacks(
            model_location=maxatac_model.results_location,
            log_location=maxatac_model.log_location,
            tensor_board_log_dir=maxatac_model.tensor_board_log_dir,
            monitor=TRAIN_MONITOR,
            reduce_lr_on_plateau=args.reduce_lr_on_plateau,
        ),
        max_queue_size=queue_size,
        use_multiprocessing=False,
        workers=1,
        verbose=1,
    )


    # Fit the model
    # logging.error("Start training the model")
    # if args.ATAC_Sampling_Multiplier == 0:
    #     training_history = maxatac_model.nn_model.fit(
    #         enq_train_gen,
    #         # model.fit() accepts a generator or Sequence that returns (inputs, targets). From the doc, when x is a generator, y should not be specified
    #         validation_data=enq_val_gen,
    #         steps_per_epoch=args.batches,
    #         validation_steps=args.batches,
    #         epochs=args.epochs,
    #         callbacks=get_callbacks(
    #             model_location=maxatac_model.results_location,
    #             log_location=maxatac_model.log_location,
    #             tensor_board_log_dir=maxatac_model.tensor_board_log_dir,
    #             monitor=TRAIN_MONITOR,
    #         ),
    #         max_queue_size=10,
    #         use_multiprocessing=False,
    #         workers=1,
    #         verbose=1,
    #     )
    # else:
    #     training_history = maxatac_model.nn_model.fit(
    #         enq_train_gen,  # model.fit() accepts a generator or Sequence that returns (inputs, targets). From the doc, when x is a generator, y should not be specified
    #         epochs=args.epochs,
    #         steps_per_epoch=steps_per_epoch_v2,
    #         validation_data=valid_data.take(
    #             (validate_examples.ROI_pool.shape[0] // args.batch_size)
    #             * args.batch_size
    #         )
    #         .cache()
    #         .repeat(args.epochs)
    #         .batch(
    #             batch_size=args.batch_size,
    #             num_parallel_calls=tensorflow.data.AUTOTUNE,
    #             drop_remainder=True,
    #         )
    #         .prefetch(tensorflow.data.AUTOTUNE),
    #         validation_steps=validate_examples.ROI_pool.shape[0] // args.batch_size,
    #         callbacks=get_callbacks(
    #             model_location=maxatac_model.results_location,
    #             log_location=maxatac_model.log_location,
    #             tensor_board_log_dir=maxatac_model.tensor_board_log_dir,
    #             monitor=TRAIN_MONITOR,
    #             reduce_lr_on_plateau=args.reduce_lr_on_plateau,
    #         ),
    #         max_queue_size=queue_size,
    #         use_multiprocessing=False,
    #         workers=1,
    #         verbose=1,
    #     )


    logging.error("Plot and save results")

    # Select best model
    best_epoch = model_selection(
        training_history=training_history, output_dir=maxatac_model.output_directory
    )

    # If plot then plot the model structure and training metrics
    # If model is transformer then also plot all the attention weights and final positional encoding
    if args.plot:
        tf = maxatac_model.train_tf
        TCL = "_".join(maxatac_model.cell_types)
        ARC = args.arch
        RR = args.rand_ratio

        export_binary_metrics(
            training_history, tf, RR, ARC, maxatac_model.results_location, best_epoch
        )
        # export_model_structure(maxatac_model.nn_model, maxatac_model.results_location)

        # data_sample = tensorflow.expand_dims(input_batch[0], axis=0)
        # if not(USE_RPE):
        #    mha_names = [f"Encoder_{i}_softmax_att_weights" for i in range(NUM_MHA)]
        # else:
        #    mha_names = [f"Transformer_block_{i}" for i in range(NUM_MHA)]
        # plot_attention_weights(maxatac_model.nn_model, mha_names, data_sample, num_heads=NUM_HEADS, file_location=args.output)

    logging.error("Results are saved to: " + maxatac_model.results_location)

    # Measure End Time of Training
    stopTime = timeit.default_timer()
    totalTime = stopTime - startTime

    # Output running time in a nice format.
    mins, secs = divmod(totalTime, 60)
    hours, mins = divmod(mins, 60)

    logging.error("Total training time: %d:%d:%d.\n" % (hours, mins, secs))

    sys.exit()
