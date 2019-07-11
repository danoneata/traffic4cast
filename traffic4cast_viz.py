#!/usr/bin/env python3

import argparse
import datetime
import sys

import src.dataset as dataset
import src.visualization as viz


def dsipatch_movie(args: argparse.Namespace):
    """ Execute "movie" sub-command.

        Args:
            args: Command line arguments.
    """

    sample = dataset.Traffic4CastSample(args.input, args.city)
    sample.load()
    viz.movie(sample, not args.no_split_channels, args.maximize)


def dsipatch_hist(args: argparse.Namespace):
    """ Execute "hist" sub-command.

        Args:
            args: Command line arguments.
    """

    sample = dataset.Traffic4CastSample(args.input, args.city)
    sample.load()

    time_locations = []
    for time_str in args.times.split(','):
        time = datetime.datetime.strptime(time_str, "%H:%M")
        time_locations.append(
            datetime.timedelta(hours=time.hour, minutes=time.minute))

    viz.hist(sample, time_locations)


def dispatch_target(args: argparse.Namespace):
    """ Execute "target" sub-command

        Args:
            args: Command line arguments.
    """
    sample = dataset.Traffic4CastSample(args.input, args.city)
    sample.load()

    viz.target(sample, (args.x, args.y))


def create_arg_parsers() -> argparse.ArgumentParser:
    """ Creates and argument parser for the traffic4cast_viz module

        Returns:
            parser : Set up argument parser.
    """

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help="display type")

    movie_parser = subparsers.add_parser('movie', help="display movie")
    movie_parser.add_argument("input", help="path of the *.hdf5 file")
    movie_parser.add_argument("-m",
                              "--maximize",
                              help="maximize display window.",
                              action="store_true")
    movie_parser.add_argument("--no-split-channels",
                              help="do not display channels separately",
                              action="store_true")
    movie_parser.add_argument("--city",
                              help="name of the city for display pourpuses",
                              default="Unknown")
    movie_parser.set_defaults(func=dsipatch_movie)

    hist_parser = subparsers.add_parser(
        'hist', help="display historgrams at specified time points")
    hist_parser.add_argument("input", help="path of the *.hdf5 file")
    hist_parser.add_argument(
        "times",
        help=
        "comma separated list of times with format %%H:%%M. e.g: 10:40,20:09",
        type=str)
    hist_parser.add_argument("--city",
                             help="name of the city for display pourpuses",
                             default="Unknown")
    hist_parser.set_defaults(func=dsipatch_hist)

    target_parser = \
        subparsers.add_parser('target',
                              help='display graphs for a target location')
    target_parser.add_argument("input", help="path of the *.hdf5 file")
    target_parser.add_argument("x",
                               help="x coordinate",
                               type=int)
    target_parser.add_argument("y",
                               help="y coordinate",
                               type=int)
    target_parser.add_argument("--city",
                               help="name of the city for display purposes",
                               default="Unknown")
    target_parser.set_defaults(func=dispatch_target)

    return parser


def main():
    """ Main entrypoint """

    parser = create_arg_parsers()
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    else:
        args = parser.parse_args()
        args.func(args)


if __name__ == "__main__":
    main()
