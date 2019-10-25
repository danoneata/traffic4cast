import argparse
import pdb
import numpy as np

import plotly
import plotly.graph_objs as go
import plotly.io as pio

from plotly.offline import init_notebook_mode, iplot, plot

import hpbandster.core.result as hpres
from hypertune import WORKERS

np.set_printoptions(precision=3, floatmode="fixed")
np.set_printoptions(suppress=True)
np.core.arrayprint._line_width = 80


def get_values(runs, id2config):
    """Parse values from hyper-parameter results to lists. Needed for parallel
    coordinates plot.

    """
    values = {}
    values["loss"] = []
    max_loss = 99999

    for run in runs:
        if run.loss is None:
            run.loss = max_loss
        values["loss"].append(run.loss)

        config = id2config[run.config_id]["config"]

        for param in config:
            if not param in values:
                values[param] = []
            values[param].append(config[param])

    return values


def data_par_coords(Worker, values, columns, KWARGS):
    data = []
    for col in columns:
        datum = {"values": values[col]}
        datum.update({"label": col.split(":")[-1]})
        try:
            datum.update(KWARGS[col])
        except KeyError:
            pass
        data.append(datum)
    return data


def main():
    """
    - load data from dir
    - print some stuff
    - generate and save par plot into temp-plot.html
    """
    parser = argparse.ArgumentParser(
        description="Plotting hyper-parameter tuning results"
    )
    parser.add_argument("-m", "--model-type", type=str, choices=WORKERS)
    parser.add_argument("--dir", type=str, help="A directory to load results from")
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

    print("A total of %i unique configurations where sampled." % len(id2config.keys()))
    print("A total of %i runs where executed." % len(all_runs))

    Worker = WORKERS[args.model_type]

    values = get_values(all_runs, id2config)

    config_space = Worker.get_configspace()
    all_columns = [
        h.name
        for h in config_space.get_hyperparameters()
        if not h.name.split(":")[-1].startswith("filt_1x1_params")
    ]
    all_columns.append("loss")

    losses = values["loss"]
    min_loss = min(losses)
    max_loss = max(losses)
    print(min_loss)

    range_max = sorted(losses)[5]

    Δ_loss = max_loss - min_loss
    KWARGS = {
        "loss": {
            "label": "loss",
            "range": [min_loss - 0.001 * Δ_loss, min_loss + 0.1 * Δ_loss],
            "constraintrange": [min_loss - 0.001 * Δ_loss, range_max],
        },
        "model:biases_type.loctime": {"label": "bias"},
        "model:biases_type.location": {"label": "bias"},
        "model:biases_type.month": {"label": "bias"},
        "model:biases_type.weekday": {"label": "bias"},
        "model:temp_reg_params.activation": {"label": "activation"},
        "model:temp_reg_params.history": {"label": "history"},
        "model:temp_reg_params.kernel_size": {"label": "kernel"},
        "model:temp_reg_params.n_layers": {"label": "n. layers"},
        "model:temp_reg_params.n_channels": {"label": "n. channels"},
    }

    def rename_labels(k):
        if k in "LxT L+T MxT WxT".split():
            return k.replace("T", "H")
        else:
            return k

    def remap(xs):
        m = {v: i for i, v in enumerate(sorted(set(xs)))}
        return [m[v] for v in xs]

    # Special columns – map them to numeric values
    special_columns = [
        "model:biases_type.loctime",
        "model:biases_type.location",
        "model:biases_type.month",
        "model:biases_type.weekday",
        "model:temp_reg_params.activation",
    ]
    for k in special_columns:
        if k not in values:
            continue
        d = {
            "tickvals": list(range(len(values[k]))),
            "ticktext": list(map(rename_labels, sorted(set(values[k])))),
        }
        try:
            KWARGS[k].update(d)
        except KeyError:
            KWARGS[k] = d
        values[k] = remap(values[k])

    data = data_par_coords(Worker, values, all_columns, KWARGS)

    parcoord_data = [
        go.Parcoords(
            line=dict(
                color="blue"
                # colorscale = 'Jet',
                # showscale = True,
                # reversescale = True,
                # cmin = 0,
                # cmax = 1
            ),
            labelfont=dict(size=12),
            tickfont=dict(size=12),
            dimensions=list(data),
        )
    ]

    plotly.offline.plot(
        parcoord_data,
        auto_open=False,
        image_filename="testfig",
        image="svg",
        image_height=300,
        image_width=750,
    )


if __name__ == "__main__":
    main()
