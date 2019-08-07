import argparse
import os
import pdb

from functools import partial

from itertools import cycle

import numpy as np

import torch

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import MSELoss
from torch.utils.data import DataLoader, Subset

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.handlers import EarlyStopping, ModelCheckpoint
from ignite.metrics import Loss

from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger, OutputHandler

from src.dataset import Traffic4CastSample, Traffic4CastDataset

from utils import sliding_window

from models import MODELS

from evaluate import CHANNELS, CITIES, EVALUATION_FRAMES, ROOT

DEBUG = os.environ.get("DEBUG", "")

MAX_EPOCHS = 64
PATIENCE = 8
LR_REDUCE_PARAMS = {
    "factor": 0.2,
    "patience": 4,
}


def select_channel(data, channel, layout):
    c = layout.find('C')
    i = Traffic4CastSample.channel_to_index[channel]
    return data.narrow(c, i, 1)


def collate_fn(history, channel, get_window, *args):
    for sample in Traffic4CastDataset.collate_list(*args):
        for window in get_window(sample):
            window_layout = "B" + sample.layout
            t = window_layout.find('T')

            batch = select_channel(window, channel, window_layout)
            batch = batch.float() #.cuda()
            assert batch.shape[-1] == 1
            batch = batch.squeeze()
            assert len(batch.shape) == 4

            tr_batch = batch.narrow(t, 0, history)
            te_batch = batch.narrow(t, history, 1)

            print(sample.date, end=" ")
            return tr_batch, te_batch


def train(city,
          model_type,
          hyper_params,
          max_epochs=MAX_EPOCHS,
          callbacks={},
          verbose=0):

    model_name = f"{model_type}_{city}"
    model_path = f"output/models/{model_name}.pth"

    train_dataset = Traffic4CastDataset(ROOT, "training", cities=[city])
    valid_dataset = Traffic4CastDataset(ROOT, "validation", cities=[city])

    print(hyper_params)

    def get_hyper_params(t):
        """Selects hyper-parameters whose key starts with `t`"""
        SEP = "_"  # separator
        get_first = lambda s: s.split(SEP)[0]
        remove_first = lambda s: SEP.join(s.split(SEP)[1:])
        return {
            remove_first(k): v
            for k, v in hyper_params.items()
            if get_first(k) == t
        }

    model = MODELS[model_type](**get_hyper_params("model"))
    # model.cuda()

    history = model.history
    channel = model.channel.capitalize()

    TRAIN_BATCH_SIZE = 32
    VALID_BATCH_SIZE = len(EVALUATION_FRAMES)

    TO_PREDICT = 1  # frame
    end_frames = [frame + TO_PREDICT for frame in EVALUATION_FRAMES]

    get_window_train = lambda sample: sample.random_temporal_batches(1, TRAIN_BATCH_SIZE, history + TO_PREDICT)
    get_window_valid = lambda sample: sample.selected_temporal_batches(VALID_BATCH_SIZE, history + TO_PREDICT, end_frames)

    collate_fn1 = partial(collate_fn, history, channel)

    collate_fn_train = partial(collate_fn1, get_window_train)
    collate_fn_valid = partial(collate_fn1, get_window_valid)

    if DEBUG:
        idxs = [1, 2, 3, 4, 5]
        train_dataset = Subset(train_dataset, idxs)
        valid_dataset = Subset(valid_dataset, idxs)

    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        collate_fn=collate_fn_train,
        shuffle=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=1,
        collate_fn=collate_fn_valid,
        shuffle=False,
    )

    optimizer = torch.optim.Adam(model.parameters(),
                                 **get_hyper_params("optimizer"))
    loss = MSELoss()

    trainer = create_supervised_trainer(model, optimizer, loss)
    evaluator = create_supervised_evaluator(model, metrics={'loss': Loss(loss)})

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(trainer):
        print("Epoch {:3d} Train loss: {:8.2f}".format(trainer.state.epoch,
                                                       trainer.state.output))

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_loss(trainer):
        evaluator.run(valid_loader)
        metrics = evaluator.state.metrics
        print("Epoch {:3d} Valid loss: {:8.2f} ←".format(
            trainer.state.epoch, metrics['loss']))

    def score_function(engine):
        return -engine.state.metrics['loss']

    if "learning-rate-scheduler" in callbacks:
        lr_reduce = ReduceLROnPlateau(optimizer, verbose=verbose, **LR_REDUCE_PARAMS)

        @evaluator.on(Events.COMPLETED)
        def update_lr_reduce(engine):
            loss = engine.state.metrics['loss']
            lr_reduce.step(loss)

    if "early-stopping" in callbacks:
        early_stopping_handler = EarlyStopping(patience=PATIENCE, score_function=score_function, trainer=trainer)
        evaluator.add_event_handler(Events.EPOCH_COMPLETED, early_stopping_handler)

    if "model-checkpoint" in callbacks:
        checkpoint_handler = ModelCheckpoint("output/models/checkpoints", model_name, score_function=score_function, n_saved=5, require_empty=False, create_dir=True)
        evaluator.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler, {"model": model})

    if "tensorboard" in callbacks:
        tensorboard_logger = TensorboardLogger(log_dir=f"output/tensorboard/{model_name}")
        tensorboard_logger.attach(trainer, log_handler=OutputHandler( tag="training", output_transform=lambda loss: {'loss': loss}), event_name=Events.ITERATION_COMPLETED)
        tensorboard_logger.attach(evaluator, log_handler=OutputHandler(tag="validation", metric_names=["loss"], another_engine=trainer), event_name=Events.EPOCH_COMPLETED)

    trainer.run(train_loader, max_epochs=max_epochs)

    if "save-model" in callbacks:
        torch.save(model.state_dict(), model_path)
        print("Model saved at:", model_path)

    return {
        'loss': evaluator.state.metrics['loss'], # HpBandSter always minimizes!
        'info': {
            'city': city,
            'model': model_type,
            'batch_size': TRAIN_BATCH_SIZE,
        },
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate a given model")
    parser.add_argument("-m",
                        "--model",
                        type=str,
                        required=True,
                        choices=MODELS,
                        help="which model to use")
    parser.add_argument("-c",
                        "--city",
                        required=True,
                        choices=CITIES,
                        help="which city to evaluate")
    parser.add_argument("-v",
                        "--verbose",
                        action="count",
                        help="verbosity level")
    args = parser.parse_args()

    print(args)

    hyper_params = {
        "optimizer_lr": 0.04,
    }

    callbacks = {
        "learning-rate-scheduler",
        "early-stopping",
        "model-checkpoint",
        "tensorboard"
        "save-model",
    }

    train(args.city,
          args.model,
          hyper_params,
          callbacks=callbacks,
          verbose=args.verbose)


if __name__ == "__main__":
    main()
