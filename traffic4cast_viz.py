#!/usr/bin/env python3
import argparse

import src.dataset as dataset
import src.visualization as viz


def create_arg_parsers() -> argparse.ArgumentParser:
    """ Creates and argument parser for the traffic4cast_viz module

        Returns:
            parser : Set up argument parser.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="path of the *.hdf5 file")
    parser.add_argument("--city",
                        help="name of the city for display pourpuses",
                        default="Unknown")
    parser.add_argument("-m",
                        "--maximize",
                        help="maximize display window.",
                        action="store_true")
    parser.add_argument("--no-split-channels",
                        help="do not display channels separately",
                        action="store_true")
    return parser


def main():
    """ Main entrypoint """
    args = create_arg_parsers().parse_args()

    sample = dataset.Traffic4CastSample(args.input, args.city)
    sample.load()
    viz.traffic4cast_show_movie(sample, not args.no_split_channels,
                                args.maximize)


if __name__ == "__main__":
    main()
