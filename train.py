import argparse
import json
import os
import os.path
import pdb
import sys

from functools import partial

from itertools import cycle

import numpy as np

import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data

import ignite.engine as engine
import ignite.handlers
import ignite.contrib.handlers.tensorboard_logger as tensorboard_logger

import src.dataset

import utils

from models import MODELS
from models.nn import ignite_selected

from evaluate import ROOT


SEED = 1337
MAX_EPOCHS = 128
PATIENCE = 4

CALLBACKS = [
    "learning-rate-scheduler",
    "early-stopping",
    "model-checkpoint",
    "tensorboard",
    "save-model",
]

LR_REDUCE_PARAMS = {
    "factor": 0.2,
    "patience": 2,
}

torch.manual_seed(SEED)


def filter_dict(d, q):
    """Filter dictionary by selecting keys starting with `q`"""
    SEP = ":"  # separator
    fst = lambda s: s.split(SEP)[0]
    rm1 = lambda s: SEP.join(s.split(SEP)[1:])
    return {rm1(k): v for k, v in d.items() if fst(k) == q}


def train(args, hyper_params):

    print(args)
    print(hyper_params)

    args.channels.sort(key=lambda x: src.dataset.Traffic4CastSample.channel_to_index[x])

    model = MODELS[args.model_type](**filter_dict(hyper_params, "model"))
    slice_size = model.past + model.future

    assert model.future == 3

    if args.model is not None:
        model_path = args.model
        model_name = os.path.basename(args.model)
        model.load(model_path)
    else:
        model_name = f"{args.model_type}_" + "_".join(args.channels +
                                                      args.cities)
        model_path = f"output/models/{model_name}.pth"

    if model.num_channels != len(args.channels):
        print(f"ERROR: Model to channels missmatch. Model can predict "
              f"{model.num_channels} channels. {len(args.channels)} were "
              "selected.")
        sys.exit(1)

    transforms = [
        lambda x: x.float(),
        lambda x: x / 255,
        src.dataset.Traffic4CastSample.Transforms.Permute("TCHW"),
        src.dataset.Traffic4CastSample.Transforms.SelectChannels(args.channels),
    ]
    train_dataset = src.dataset.Traffic4CastDataset(ROOT, "training",
                                                    args.cities, transforms)
    valid_dataset = src.dataset.Traffic4CastDataset(ROOT, "validation",
                                                    args.cities, transforms)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,
        collate_fn=src.dataset.Traffic4CastDataset.collate_list,
        shuffle=True)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=1,
        collate_fn=src.dataset.Traffic4CastDataset.collate_list,
        shuffle=False)

    ignite_train = ignite_selected(
        train_loader,
        slice_size=slice_size,
        **filter_dict(hyper_params, "ignite_selected"),
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        **filter_dict(hyper_params, "optimizer"),
    )
    loss = nn.MSELoss()

    best_loss = 1.0

    device = args.device
    if device.find('cuda') != -1 and not torch.cuda.is_available():
        device = 'cpu'
    trainer = engine.create_supervised_trainer(model,
                                               optimizer,
                                               loss,
                                               device=device,
                                               prepare_batch=model.ignite_batch)
    evaluator = engine.create_supervised_evaluator(
        model,
        metrics={'loss': ignite.metrics.Loss(loss)},
        device=device,
        prepare_batch=model.ignite_batch)

    @trainer.on(engine.Events.ITERATION_COMPLETED)
    def log_training_loss(trainer):
        print("Epoch {:3d} Train loss: {:8.6f}".format(trainer.state.epoch,
                                                       trainer.state.output))

    @trainer.on(engine.Events.EPOCH_COMPLETED)
    def log_validation_loss(trainer):
        evaluator.run(ignite_selected(valid_loader, slice_size=slice_size))
        metrics = evaluator.state.metrics
        print("Epoch {:3d} Valid loss: {:8.6f} ←".format(
            trainer.state.epoch, metrics['loss']))
        trainer.state.dataloader = ignite_selected(
            train_loader,
            slice_size=slice_size,
            **filter_dict(hyper_params, "ignite_selected"))
        nonlocal best_loss
        best_loss = min(best_loss, metrics['loss'])

    if "learning-rate-scheduler" in args.callbacks:
        lr_reduce = lr_scheduler.ReduceLROnPlateau(optimizer,
                                                   verbose=args.verbose,
                                                   **LR_REDUCE_PARAMS)
        @evaluator.on(engine.Events.COMPLETED)
        def update_lr_reduce(engine):
            loss = engine.state.metrics['loss']
            lr_reduce.step(loss)

    def score_function(engine):
        return -engine.state.metrics['loss']

    if "early-stopping" in args.callbacks:
        early_stopping_handler = ignite.handlers.EarlyStopping(
            patience=PATIENCE, score_function=score_function, trainer=trainer)
        evaluator.add_event_handler(engine.Events.EPOCH_COMPLETED,
                                    early_stopping_handler)

    if "model-checkpoint" in args.callbacks:
        checkpoint_handler = ignite.handlers.ModelCheckpoint(
            "output/models/checkpoints",
            model_name,
            score_function=score_function,
            n_saved=1,
            require_empty=False,
            create_dir=True)
        evaluator.add_event_handler(engine.Events.EPOCH_COMPLETED,
                                    checkpoint_handler, {"model": model})

    if "tensorboard" in args.callbacks:
        logger = tensorboard_logger.TensorboardLogger(
            log_dir=f"output/tensorboard/{model_name}")
        logger.attach(trainer,
                      log_handler=tensorboard_logger.OutputHandler(
                          tag="training",
                          output_transform=lambda loss: {'loss': loss}),
                      event_name=engine.Events.ITERATION_COMPLETED)
        logger.attach(evaluator,
                      log_handler=tensorboard_logger.OutputHandler(
                          tag="validation",
                          metric_names=["loss"],
                          another_engine=trainer),
                      event_name=engine.Events.EPOCH_COMPLETED)

    trainer.run(ignite_train, **filter_dict(hyper_params, "trainer_run"))

    if "save-model" in args.callbacks and not "model-checkpoint" in args.callbacks:
        torch.save(model.state_dict(), model_path)
        print("Model saved at:", model_path)
    elif "save-model" in args.callbacks:
        # Move best model from checkpoint directory to output/models
        checkpoints_dir = "output/models/checkpoints"
        source, *_ = [
            f for f in reversed(utils.sorted_ls(checkpoints_dir))
            if f.startswith(model_name)
        ]  # get most recent model
        os.rename(os.path.join(checkpoints_dir, source), model_path)
        print("Model saved at:", model_path)

    return {
        'loss': best_loss, # HpBandSter always minimizes!
        'info': {
            'args': vars(args),
            'hyper-params': hyper_params,
        },
    }


def get_train_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model-type",
                        type=str,
                        required=True,
                        choices=MODELS,
                        help="which model type to train")
    parser.add_argument("-c",
                        "--cities",
                        nargs='+',
                        required=True,
                        help="which cities to train on")
    parser.add_argument("--channels",
                        nargs='+',
                        help="List of channels to use.")
    parser.add_argument("-m",
                        "--model",
                        type=str,
                        default=None,
                        required=False,
                        help="path to model to load")
    parser.add_argument(
        "-d",
        "--device",
        required=False,
        default='cuda',
        choices=['cpu', 'cuda', *[f"cuda:{n}" for n in range(8)]],
        type=str,
        help=("which device to use. defaults to current cuda "
              "device if present otherwise to current cpu"))
    parser.add_argument("--callbacks",
                        required=False,
                        nargs='+',
                        choices=CALLBACKS,
                        default=CALLBACKS,
                        help="what action to perform during training")
    parser.add_argument("-v",
                        "--verbose",
                        action="count",
                        help="verbosity level")
    return parser


def main():
    parser = argparse.ArgumentParser(
        parents=[get_train_parser()],
        description="Evaluate a given model",
    )

    # Extra options for training
    def epoch_fraction(fraction):
        try:
            fraction = float(fraction)
        except ValueError:
            raise argparse.ArgumentTypeError(f"Must be floating point.")
        if (fraction <= 0 or fraction > 1.0):
            raise argparse.ArgumentTypeError(f"Must be in (0, 1.0]")
        else:
            return fraction

    parser.add_argument("--epoch-fraction",
                        required=False,
                        default=0.2,
                        type=epoch_fraction,
                        help=("fraction of the training set to use each epoch."
                              "Value must be in (0, 1.0] Default: 0.2."
                              "At least one sample will be used if the fraction"
                              "is less than a sample."))
    parser.add_argument("--minibatch-size",
                        required=False,
                        default=32,
                        type=int,
                        help="mini batch size. Default: 32")
    parser.add_argument("--num-minibatches",
                        default=16,
                        type=int,
                        help="number of minibatches per sample. Default: 16")
    parser.add_argument(
        "--hyper-params",
        required=False,
        help=(
            "path to JSON file containing hyper-parameter configuration "
            "(over-writes other hyper-parameters passed through the "
            "command line)."),
    )
    args = parser.parse_args()

    hyper_params = {
        "optimizer:lr": 0.01,
        "trainer_run:max_epochs": MAX_EPOCHS,
        "ignite_selected:epoch_fraction": args.epoch_fraction,
        # "ignite_random:minibatch_size": args.minibatch_size,
        # "ignite_random:num_minibatches": args.num_minibatches,
    }

    if args.hyper_params and os.path.exists(args.hyper_params):
        with open(args.hyper_params, "r") as f:
            hyper_params1 = json.load(f)
        hyper_params.update(hyper_params1)

    train(args, hyper_params)


if __name__ == "__main__":
    main()
