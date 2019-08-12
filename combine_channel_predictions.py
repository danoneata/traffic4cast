import argparse
import os
import pdb

import h5py
import numpy as np

import evaluate
import src.dataset
import submission_write


def main():
    parser = argparse.ArgumentParser(description="Combines predictions of single-channel models.")
    parser.add_argument(
        "-s", "--split",
        default="validation",
        choices={"validation", "test"},
        required=True,
        help="data split",
    )
    parser.add_argument(
        "-c", "--city",
        choices=evaluate.CITIES,
        required=True,
        help="which city to evaluate",
    )
    parser.add_argument(
        "--volume",
        required=True,
        help="model name for volume",
    )
    parser.add_argument(
        "--speed",
        required=True,
        help="model name for speed",
    )
    parser.add_argument(
        "--heading",
        required=True,
        help="model name for heading",
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        type=str,
        help="name of the combined model",
    )
    args = parser.parse_args()
    args_dict = vars(args)
    print(args)

    def select_channel(data, channel):
        # data has shape (N, T, W, H, C)
        i = src.dataset.Traffic4CastSample.channel_to_index[channel.capitalize()]
        try:
            return data[:, :, :, :, i]
        except:
            pdb.set_trace()

    load_data = lambda path: np.array(h5py.File(path, 'r')['array'])
    folders = {
        c: evaluate.get_prediction_folder(args.split, args_dict[c], args.city)
        for c in evaluate.CHANNELS
    }
    filenames = {c: os.listdir(folder) for c, folder in folders.items()}
    contain_same_files = (
        set(filenames['heading']) ==
        set(filenames['speed']) ==
        set(filenames['volume']))
    assert contain_same_files, (
        "The models don't have predictions for the same files")
    filenames = filenames['heading']

    folder_output = evaluate.get_prediction_folder(args.split, args.output, args.city)
    os.makedirs(folder_output, exist_ok=True)

    for filename in filenames:
        print(filename)
        data = {c: load_data(os.path.join(folders[c], filename)) for c in evaluate.CHANNELS}
        data_output = [select_channel(data[c], c) for c in evaluate.CHANNELS]
        data_output = np.stack(data_output, axis=-1)
        data_output = data_output.astype(np.uint8)
        submission_write.write_data(data_output, os.path.join(folder_output, filename))


if __name__ == '__main__':
    main()
