import argparse
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

from models import MODELS

from evaluate import ROOT

SEED = 1337
MAX_EPOCHS = 128
PATIENCE = 4
LR_REDUCE_PARAMS = {
    "factor": 0.2,
    "patience": 2,
}


torch.manual_seed(SEED)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a given model")
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
                        default=0.5,
                        type=epoch_fraction,
                        help=("fraction of the training set to use each epoch."
                              "Value must be in (0, 1.0] Default: 0.5."
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
        "-d",
        "--device",
        required=False,
        default='cuda',
        choices=['cpu', 'cuda', *[f"cuda:{n}" for n in range(8)]],
        type=str,
        help=("which device to use. defaults to current cuda "
              "device if present otherwise to current cpu"))
    parser.add_argument("--no-log-tensorboard",
                        required=False,
                        default=False,
                        action='store_true',
                        help="do not log to tensorboard format. Default false.")
    parser.add_argument("-v",
                        "--verbose",
                        action="count",
                        help="verbosity level")
    args = parser.parse_args()
    args.channels.sort(
        key=lambda x: src.dataset.Traffic4CastSample.channel_to_index[x])

    print(args)

    model = MODELS[args.model_type]()
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

    ignite_train = model.ignite_random(train_loader, args.num_minibatches,
                                       args.minibatch_size, args.epoch_fraction)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.04)

    mse = nn.MSELoss()
    bce = nn.BCELoss()

    def tr_loss(inp, tgt):
        out, mask, y = inp
        B, F, C, H, W = out.shape
        tgt = tgt.view(B, F, C, H, W)
        nnzs = (tgt[:, :, :1] > 0)
        idxs = nnzs.float()
        nnzs = nnzs.repeat(1, 1, 3, 1, 1)
        loss1 = mse(out, tgt)
        loss2 = bce(mask, idxs)
        loss3 = mse(y[nnzs], tgt[nnzs])
        # loss4 = mse(idxs * y, tgt)  # predict perfectly the missing values
        # loss5 = mse(mask * tgt, tgt)  # predict perfectly the values
        # loss6 = mse(idxs * tgt, tgt) # zero
        losses = [
            '{:.6f}'.format(loss.detach().cpu().numpy())
            for loss in [
                loss1,
                loss2,
                loss3,
                # loss4,
                # loss5,
            ]
        ]
        print(*losses)
        return (
            1.0 * loss1 +
            0.1 * loss2 +
            0.1 * loss3
        )
    def te_loss(inp, tgt):
        out, *_ = inp
        B, F, C, H, W = out.shape
        tgt = tgt.view(B, F, C, H, W)
        return mse(out, tgt)

    # tr_loss = mse
    # te_loss = mse

    device = args.device
    if device.find('cuda') != -1 and not torch.cuda.is_available():
        device = 'cpu'
    trainer = engine.create_supervised_trainer(model,
                                               optimizer,
                                               tr_loss,
                                               device=device,
                                               prepare_batch=model.ignite_batch)
    evaluator = engine.create_supervised_evaluator(
        model,
        metrics={'loss': ignite.metrics.Loss(te_loss)},
        device=device,
        prepare_batch=model.ignite_batch)

    # @trainer.on(engine.Events.ITERATION_COMPLETED)
    # def log_training_loss(trainer):
    #     print("Epoch {:3d} Train loss: {:8.6f}".format(trainer.state.epoch,
    #                                                    trainer.state.output))

    @trainer.on(engine.Events.EPOCH_COMPLETED)
    def log_validation_loss(trainer):
        evaluator.run(model.ignite_all(valid_loader, args.minibatch_size))
        metrics = evaluator.state.metrics
        print("Epoch {:3d} Valid loss: {:8.6f} ←".format(
            trainer.state.epoch, metrics['loss']))
        trainer.state.dataloader = model.ignite_random(train_loader,
                                                       args.num_minibatches,
                                                       args.minibatch_size,
                                                       args.epoch_fraction)

    lr_reduce = lr_scheduler.ReduceLROnPlateau(optimizer,
                                               verbose=args.verbose,
                                               **LR_REDUCE_PARAMS)

    @evaluator.on(engine.Events.COMPLETED)
    def update_lr_reduce(engine):
        loss = engine.state.metrics['loss']
        lr_reduce.step(loss)

    def score_function(engine):
        return -engine.state.metrics['loss']

    early_stopping_handler = ignite.handlers.EarlyStopping(
        patience=PATIENCE, score_function=score_function, trainer=trainer)
    evaluator.add_event_handler(engine.Events.EPOCH_COMPLETED,
                                early_stopping_handler)

    checkpoint_handler = ignite.handlers.ModelCheckpoint(
        "output/models/checkpoints",
        model_name,
        score_function=score_function,
        n_saved=1,
        require_empty=False,
        create_dir=True)
    evaluator.add_event_handler(engine.Events.EPOCH_COMPLETED,
                                checkpoint_handler, {"model": model})

    if not args.no_log_tensorboard:
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

    trainer.run(ignite_train, max_epochs=MAX_EPOCHS)
    # torch.save(model.state_dict(), model_path)
    # print("Model saved at:", model_path)


if __name__ == "__main__":
    main()
