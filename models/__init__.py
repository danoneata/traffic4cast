import models.baseline
import models.nn

MODELS = {
    "naive": models.baseline.Naive,
    "zeros": models.baseline.Zeros,
    "single-channel-temporal-regression-12": lambda: models.nn.Temporal(12, 1, 1, models.nn.TemporalRegression(12)),
    "seasonal-temporal-regression": lambda: models.nn.TemporalDate(12, 3, 1, models.nn.SeasonalTemporalRegression(12, 3)),
    "seasonal-temporal-regression-heading": lambda: models.nn.TemporalDate(12, 3, 1, models.nn.SeasonalTemporalRegressionHeading(12, 3)),
    "calba": lambda history=12, n_layers=3, n_channels=16: models.nn.TemporalDate(history, 3, 1, models.nn.Calba(history, 3, n_layers, n_channels)),
}
