from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    CSVLogger,
    TensorBoard,
    ReduceLROnPlateau,
    Callback,
)
from maxatac.utilities.constants import DEFAULT_ADAM_LEARNING_RATE


def get_callbacks(
    model_location,
    log_location,
    tensor_board_log_dir,
    monitor,
    save_weights_only=True,
    save_best_only=False,
    append_log=False,
    tensor_board_write_images=False,
    tensor_board_write_graph=True,
    reduce_lr_on_plateau=False,
):
    callbacks = [
        ModelCheckpoint(
            filepath=model_location,
            save_weights_only=save_weights_only,
            save_best_only=save_best_only,
            monitor=monitor,
        ),
        CSVLogger(log_location, separator=",", append=append_log),
        TensorBoard(
            tensor_board_log_dir,
            write_images=tensor_board_write_images,
            write_graph=tensor_board_write_graph,
            update_freq="batch",
        ),
    ]
    if reduce_lr_on_plateau:
        callbacks.append(
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.1,
                patience=5,  # depends on average epoch number
                min_lr=DEFAULT_ADAM_LEARNING_RATE / 100,
            )
        )
    return callbacks
