from typing import List, Dict
import datetime

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import torch

import src.dataset as dataset


def traffic4cast_show_movie(sample: dataset.Traffic4CastSample,
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

    time_step = datetime.timedelta(minutes=5)
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


def remap_heading(data: torch.tensor,
                  map: Dict = {
                      0: 0,
                      1: 1,
                      85: 2,
                      170: 3,
                      255: 4
                  }) -> torch.tensor:
    """ Remaps the values in data according to the map.

        Args:
            data: Data to remap.
            map: Value map.

        Returns:
            Tensor with remaped values.
    """

    remaped = data.clone().detach()
    for key, val in map.items():
        remaped[remaped == key] = val

    return remaped
