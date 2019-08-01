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
            batch = batch.float().cuda()
            assert batch.shape[-1] == 1
            batch = batch.squeeze()
            assert len(batch.shape) == 4

            tr_batch = batch.narrow(t, 0, history)
            te_batch = batch.narrow(t, history, 1)

            print(sample.date, end=" ")
            return tr_batch, te_batch


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

    model_name = f"{args.model}_{args.city}"
    model_path = f"output/models/{model_name}.pth"

    train_dataset = Traffic4CastDataset(ROOT, "training", cities=[args.city])
    valid_dataset = Traffic4CastDataset(ROOT, "validation", cities=[args.city])

    model = MODELS[args.model]()
    model.cuda()

    history = model.history
    channel = model.channel.capitalize()

    train_batch_size = 32
    valid_batch_size = len(EVALUATION_FRAMES)

    to_predict = 1  # frame
    end_frames = [frame + to_predict for frame in EVALUATION_FRAMES]

    get_window_train = lambda sample: sample.random_temporal_batches(1, train_batch_size, history + to_predict)
    get_window_valid = lambda sample: sample.selected_temporal_batches(valid_batch_size, history + to_predict, end_frames)

    collate_fn1 = partial(collate_fn, history, channel)

    collate_fn_train = partial(collate_fn1, get_window_train)
    collate_fn_valid = partial(collate_fn1, get_window_valid)

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

    optimizer = torch.optim.Adam(model.parameters(), lr=0.04)
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

    # Learning rate scheduler
    lr_reduce = ReduceLROnPlateau(optimizer,
                                  verbose=args.verbose,
                                  **LR_REDUCE_PARAMS)

    @evaluator.on(Events.COMPLETED)
    def update_lr_reduce(engine):
        loss = engine.state.metrics['loss']
        lr_reduce.step(loss)

    def score_function(engine):
        return -engine.state.metrics['loss']

    # Early stopping
    early_stopping_handler = EarlyStopping(patience=PATIENCE,
                                           score_function=score_function,
                                           trainer=trainer)
    evaluator.add_event_handler(Events.EPOCH_COMPLETED, early_stopping_handler)

    # Model checkpoint
    checkpoint_handler = ModelCheckpoint("output/models/checkpoints",
                                         model_name,
                                         score_function=score_function,
                                         n_saved=5,
                                         require_empty=False,
                                         create_dir=True)
    evaluator.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler,
                                {"model": model})

    # Tensorboard
    tensorboard_logger = TensorboardLogger(
        log_dir=f"output/tensorboard/{model_name}")
    tensorboard_logger.attach(trainer,
                              log_handler=OutputHandler(
                                  tag="training",
                                  output_transform=lambda loss: {'loss': loss}),
                              event_name=Events.ITERATION_COMPLETED)
    tensorboard_logger.attach(evaluator,
                              log_handler=OutputHandler(tag="validation",
                                                        metric_names=["loss"],
                                                        another_engine=trainer),
                              event_name=Events.EPOCH_COMPLETED)

    trainer.run(train_loader, max_epochs=MAX_EPOCHS)
    torch.save(model.state_dict(), model_path)
    print("Model saved at:", model_path)


if __name__ == "__main__":
    main()
