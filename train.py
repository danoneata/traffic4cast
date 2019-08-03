import argparse
import os
import pdb

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

from evaluate import CHANNELS, CITIES, SUBMISSION_FRAMES, ROOT

MAX_EPOCHS = 1
PATIENCE = 8
LR_REDUCE_PARAMS = {
    "factor": 0.2,
    "patience": 4,
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

    print(args)

    model_name = f"{args.model}_{args.city}"
    model_path = f"output/models/{model_name}.pth"

    model = MODELS[args.model]()

    transforms = [
        lambda x: x.float(),
        lambda x: x / 255,
        src.dataset.Traffic4CastSample.Transforms.Permute("TCHW"),
        src.dataset.Traffic4CastSample.Transforms.SelectChannels(
            model.channels),
    ]
    train_dataset = src.dataset.Traffic4CastDataset(ROOT, "training",
                                                    [args.city], transforms)
    valid_dataset = src.dataset.Traffic4CastDataset(ROOT, "validation",
                                                    [args.city], transforms)

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

    num_minibatches = 2
    minibatch_size = 8
    slice_size = 15
    ignite_train = model.ignite_random(train_loader, num_minibatches,
                                       minibatch_size)
    ignite_valid = model.ignite_all(valid_loader, num_minibatches)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.04)
    loss = nn.MSELoss()

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
        print("Epoch {:3d} Train loss: {:8.2f}".format(trainer.state.epoch,
                                                       trainer.state.output))

    @trainer.on(engine.Events.EPOCH_COMPLETED)
    def log_validation_loss(trainer):
        evaluator.run(ignite_valid)
        metrics = evaluator.state.metrics
        print("Epoch {:3d} Valid loss: {:8.2f} ←".format(
            trainer.state.epoch, metrics['loss']))

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
        n_saved=5,
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
    torch.save(model.state_dict(), model_path)
    print("Model saved at:", model_path)


if __name__ == "__main__":
    main()
