import datetime
from typing import List, Dict

import matplotlib.animation as animation
import matplotlib.colors as colors
import matplotlib.lines as lines
import matplotlib.pyplot as plt
import numpy as np
import torch

import src.dataset as dataset


def movie(sample: dataset.Traffic4CastSample,
          split_channels: bool = True,
          maximize: bool = False):
    """ Creates and display a matplotlib animation out of a Traffic4CastSample.

        Args:
            sample: The Traffic4CastSample.
            split_channels: Boolean indicating whether to display the cahannels
                seprate or together.
            maximize: Bolean indicatig whether to maximize the movie window
                or not.
    """

    if split_channels:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        ax1.set_title("Volume")
        ax2.set_title("Speed")
        ax3.set_title("Heading. UNEWS = WRGBY")
        direction_cmap = colors.ListedColormap(
            ['white', 'red', 'green', 'blue', 'yellow'])
    else:
        fig, ax1 = plt.subplots(1, 1)

    fig.suptitle(f"{sample.city}, {sample.date.strftime('%d %B, %Y')}")

    if maximize:
        mng = plt.get_current_fig_manager()
        mng.window.showMaximized()

    time_step = dataset.Traffic4CastSample.time_step_delta
    time_stamp_box = dict(boxstyle="round", color=(1., 0.5, 0.5))
    time_stamp_pos = (0, 20)
    frames = []
    for frame in range(sample.data.shape[0]):
        if split_channels:
            direction = remap_heading(sample.data[frame, :, :, 2])
            frames.append([
                ax1.imshow(sample.data[frame, :, :, 0], cmap='Reds'),
                ax1.text(*time_stamp_pos,
                         f"{time_step*frame}",
                         bbox=time_stamp_box),
                ax2.imshow(sample.data[frame, :, :, 1], cmap='Greens'),
                ax2.text(*time_stamp_pos,
                         f"{time_step*frame}",
                         bbox=time_stamp_box),
                ax3.imshow(direction, cmap=direction_cmap),
                ax3.text(*time_stamp_pos,
                         f"{time_step*frame}",
                         bbox=time_stamp_box),
            ])
        else:
            frames.append([
                ax1.imshow(sample.data[frame]),
                ax1.text(*time_stamp_pos,
                         f"{time_step*frame}",
                         bbox=time_stamp_box)
            ])

    animation.ArtistAnimation(fig,
                              frames,
                              interval=200,
                              blit=True,
                              repeat_delay=1000)
    plt.show()


def remap_heading(data: torch.tensor, heading_map: Dict = None) -> torch.tensor:
    """ Remaps the values in data according to the map.

        Args:
            data: Data to remap.
            heading_map: Value map. Default remaping values:
                {0:0, 1:1, 85:2, 170:3, 255:4}

        Returns:
            Tensor with remaped values.
    """

    if heading_map is None:
        heading_map = {0: 0, 1: 1, 85: 2, 170: 3, 255: 4}

    remaped = data.clone().detach()
    for key, val in heading_map.items():
        remaped[remaped == key] = val

    return remaped


def hist(sample: dataset.Traffic4CastSample,
         time_points: List[datetime.timedelta]):
    """ Displays historgrams at diffrent times.

        Creates a display of historgrams for each of the data channels at
        specified times. The display is partitioned in 3 columes, one for each
        channel, and as many rows as requested.

        The historgrams take into account only the data points where
        the value of the 'volume' channel is diffrent from 0.

        The historgrams characteristics are:
        - 'volume' channel: 256 bins ploted on a log scale.
        - 'speed' channel: 256 bins linear scale.
        - 'heading' channel: 5 bins - one for each cardinal point and one for
            the non preferentail heading value.

        Args:
            sample: The Traffic4CastSample data sample.
            time_points: A list of point in time at which to compute the
                histograms.
    """

    fig, axes = plt.subplots(len(time_points), sample.data.shape[3])
    fig.suptitle(f"{sample.city}, {sample.date.strftime('%Y, %B %d')}")
    axes = axes.flatten()
    for i in range(sample.data.shape[3]):
        axes[i].set_title(dataset.Traffic4CastSample.channel_label[i])

    for point, time in enumerate(time_points):
        frame = int(time / dataset.Traffic4CastSample.time_step_delta)
        non_zero_volume_mask = sample.data[frame, :, :, 0] != 0
        valid_data = sample.data[frame, non_zero_volume_mask]

        # Volume.
        axes[point * 3 + 0].set_ylabel(str(time), rotation=0, size='large')
        axes[point * 3 + 0].hist(valid_data[:, 0], bins=256, log=True)
        volume_stats = (
            f'min: {valid_data[:, 0].min()}', f'max: {valid_data[:, 0].min()}',
            f'mean: {valid_data[:, 0].to(torch.float, copy=True).mean() : .2f}',
            f'median: {valid_data[:, 0].median()}')
        volume_legend = [lines.Line2D([0], [0], label=l) for l in volume_stats]
        axes[point * 3 + 0].legend(handles=volume_legend)

        # Speed.
        speed_stats = (
            f'min: {valid_data[:, 1].min()}', f'max: {valid_data[:, 1].min()}',
            f'mean: {valid_data[:, 1].to(torch.float, copy=True).mean() : .2f}',
            f'median: {valid_data[:, 1].median()}')
        speed_legend = [lines.Line2D([0], [0], label=l) for l in speed_stats]
        axes[point * 3 + 1].legend(handles=speed_legend)
        axes[point * 3 + 1].hist(valid_data[:, 1], bins=256)

        # Heading.
        xticks = [1, 2, 3, 4, 5]
        axes[point * 3 + 2].bar(
            xticks,
            np.histogram(valid_data[:, 2], bins=[0, 1, 85, 170, 255, 256])[0])
        axes[point * 3 + 2].set_xticks(xticks)
        axes[point * 3 + 2].set_xticklabels(['U', 'N', 'E', 'W', 'S'])

    plt.show()
