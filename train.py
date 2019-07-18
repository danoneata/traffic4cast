import argparse
import os
import pdb

from itertools import cycle

import numpy as np

import torch

from torch.utils.data import DataLoader
from torch.nn import MSELoss

from src.dataset import Traffic4CastDataset

from utils import sliding_window

from models import MODELS

from evaluate import ROOT, CITIES


def evaluate1(model, criterion, batch):
    frames_i = batch[:, : 3 * 12]
    frames_o = batch[:, 3 * 12 :]
    pred = model.forward(frames_i)
    return criterion(pred, frames_o)


def train(model, get_loader, optimizer, is_checkpoint, callbacks=tuple()):
    # Build data loader
    data_loader_train = get_loader("train")
    data_loader_valid = get_loader("valid")

    # Loss
    criterion = MSELoss()
    # best_result = BestResult()

    def get_batch(data):
        batch = data[0].data[:15].float()
        batch = batch.permute(0, 3, 1, 2)
        batch = batch.reshape(-1, batch.shape[2], batch.shape[3])
        batch = batch.unsqueeze(0)
        return batch.cuda()

    for i, data in enumerate(cycle(data_loader_train)):

        batch = get_batch(data)

        model.train()
        loss = evaluate1(model, criterion, batch)

        model.zero_grad()
        loss.backward()
        optimizer.step()

        if is_checkpoint(i):

            model.eval()
            losses = [evaluate1(model, criterion, get_batch(d)).data for d in data_loader_valid]

            loss_valid = np.mean(losses)
            loss_train = loss.data

            state = {
                "step": i,
                "loss_train": loss_train,
                "loss_valid": loss_valid,
                # "is_best": best_result.is_best(loss_valid),
                "model": model,
            }
            # best_result.update(loss_valid)
            print(state)

            try:
                for callback in callbacks:
                    callback(**state)
            except StopIteration:
                break

    # return {"loss": best_result.best_loss, "status": STATUS_OK}


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

    train_dataset = Traffic4CastDataset(ROOT, "training", cities=[args.city])
    valid_dataset = Traffic4CastDataset(ROOT, "validation", cities=[args.city])

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=Traffic4CastDataset.collate_list)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=True, collate_fn=Traffic4CastDataset.collate_list)

    model = MODELS[args.model]()
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters())
    is_checkpoint = lambda i: i % 100 == 0

    def get_loader(split):
        if split == 'train':
            return train_loader
        else:
            return valid_loader

    train(
        model,
        get_loader,
        optimizer,
        is_checkpoint=is_checkpoint,
    )


if __name__ == "__main__":
    main()
