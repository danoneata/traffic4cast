import models.baseline
import models.nn

MODELS = {
    "naive": models.baseline.Naive,
    "zeros": models.baseline.Zeros,
    "single-channel-temporal-regression-12": lambda: models.nn.Temporal(12, 1, 1, models.nn.TemporalRegression(12)),
    "seasonal-temporal-regression": lambda: models.nn.TemporalDate(12, 3, 1, models.nn.SeasonalTemporalRegression(12, 3)),
    "seasonal-temporal-regression-heading": lambda: models.nn.TemporalDate(12, 3, 1, models.nn.SeasonalTemporalRegressionHeading(12, 3)),
    "calba": lambda history=12, n_layers=3, n_channels=16: models.nn.TemporalDate(history, 3, 1, models.nn.Calba(history, 3, n_layers, n_channels)),
    "petronius": lambda history=12: models.nn.TemporalDate(history, 3, 1, models.nn.Petronius(history)),
    "petronius-param": lambda *args, **kwargs: models.nn.TemporalDate(12, 3, 1, models.nn.PetroniusParam(*args, **kwargs)),
    "petronius-heading": lambda *args, **kwargs: models.nn.TemporalDate(12, 3, 1, models.nn.PetroniusHeading(*args, **kwargs)),
    "marcus": lambda *args, **kwargs: models.nn.TemporalDate(12, 3, 1, models.nn.Marcus(*args, **kwargs)),
    "vicinius": lambda *args, **kwargs: models.nn.TemporalDate(12, 3, 1, models.nn.Vicinius(*args, **kwargs)),
    "calina-heading": lambda *args, **kwargs: models.nn.TemporalDate(12, 3, 1, models.nn.CalinaHeading(*args, **kwargs)),
    "nero": lambda *args, **kwargs: models.nn.TemporalDate(12, 3, 3, models.nn.Nero(*args, **kwargs)),
}

for l in (4, 8, 16):
    for c in (16, 32, 64):
        k = f"nero-{l}-{c}"
        t = dict(n_channels=c, n_layers=l)
        MODELS[k] = lambda *args, **kwargs: models.nn.TemporalDate(12, 3, 3, models.nn.Nero(*args, temp_reg_params=t, **kwargs))
