import argparse
import os
import pdb

from functools import partial

from itertools import cycle

import numpy as np

import torch

from torch.utils.data import DataLoader, RandomSampler
from torch.nn import MSELoss

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Loss

from src.dataset import Traffic4CastSample, Traffic4CastDataset

from utils import sliding_window

from models import MODELS

from evaluate import ROOT, CITIES, CHANNELS


def select_channel(data, channel):
    # Every third frame belongs to the same channel.
    s = Traffic4CastSample.channel_to_index[channel]
    return data[:, s::3]


def collate_fn(history, channel, *args):
    for sample in Traffic4CastDataset.collate_list(*args):
        for window in sample.sliding_window_generator(history + 1, 4, 32):
            batch = select_channel(window, channel)
            tr_batch = batch[:, :history].float().cuda()
            te_batch = batch[:, history:].float().cuda()
            print(sample.date, list(tr_batch.shape), end=" ")
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

    def get_model_path():
        return f"output/models/{args.model}_{args.city}.pth"

    train_dataset = Traffic4CastDataset(ROOT, "training", cities=[args.city])
    valid_dataset = Traffic4CastDataset(ROOT, "validation", cities=[args.city])

    model = MODELS[args.model]()
    model.cuda()

    collate_fn1 = partial(collate_fn, model.history, model.channel.capitalize())
    train_loader = DataLoader(train_dataset,
                              batch_size=1,
                              collate_fn=collate_fn1,
                              shuffle=True)
    valid_loader = DataLoader(valid_dataset,
                              batch_size=1,
                              collate_fn=collate_fn1,
                              shuffle=False)

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

    trainer.run(train_loader, max_epochs=16)
    torch.save(model.state_dict(), get_model_path())
    print("Model saved at:", get_model_path())


if __name__ == "__main__":
    main()
