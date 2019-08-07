import argparse
import numpy as np

import plotly
import plotly.graph_objs as go

import hpbandster.core.result as hpres
from hypertune import PyTorchWorker as worker

np.set_printoptions(precision=3, floatmode='fixed')
np.set_printoptions(suppress=True)
np.core.arrayprint._line_width = 80


def get_values(runs, id2config):
    """
		Parse values from hyper-parameter results to lists. 
		Needed for parallel coordinates plot.
	"""
    values = {}
    values["loss"] = []
    max_loss = 99999

    for run in runs:
        if run.loss is None:
            run.loss = max_loss
        values["loss"].append(run.loss)

        config = id2config[run.config_id]['config']

        for param in config:
            if not param in values:
                values[param] = []
            values[param].append(config[param])

    return values


def data_par_coords(values, columns):
    config_space = worker.get_configspace()
    # print(config_space)
    # print(config_space.get_hyperparameter("optimizer_lr"))

    KWARGS = {
        "optimizer_lr": {
            "label": 'Learning rate',
            "range": [1e-6, 1e-1]
        },
        "loss": {
            "label": 'Loss'
        }
    }

    data = []
    for col in columns:
        try:
            datum = {"values": values[col]}
            datum.update(KWARGS[col])
            data.append(datum)
        except KeyError:
            pass
    return data


def main():
    """
		load data from dir
		print some stuff
		generate and save par plot into temp-plot.html
	"""
    parser = argparse.ArgumentParser(
        description='Plotting hyper-parameter tuning results')
    parser.add_argument('--dir',
                        type=str,
                        help='A directory to load results from')
    args = parser.parse_args()

    if args.dir is None:
        print("usage: python hypetune_results.py --dir DIR")
        return

    res = hpres.logged_results_to_HBS_result("./" + args.dir)
    id2config = res.get_id2config_mapping()
    incumbent = res.get_incumbent_id()
    # print(id2config[incumbent])
    all_runs = res.get_all_runs()
    print(all_runs[0])

    print('A total of %i unique configurations where sampled.' %
          len(id2config.keys()))
    print('A total of %i runs where executed.' % len(all_runs))

    values = get_values(all_runs, id2config)
    all_columns = ['loss', 'optimizer_lr', 'blabla']
    data = data_par_coords(values, all_columns)

    parcoord_data = [
        go.Parcoords(
            line=dict(
                color='blue'
                # colorscale = 'Jet',
                # showscale = True,
                # reversescale = True,
                # cmin = 0,
                # cmax = 1
            ),
            labelfont=dict(size=20),
            tickfont=dict(size=18),
            dimensions=list(data))
    ]

    plotly.offline.plot(parcoord_data, auto_open=False)


if __name__ == "__main__":
    main()
