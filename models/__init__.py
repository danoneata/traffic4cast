import models.baseline
import models.nn

MODELS = {
    "naive": models.baseline.Naive,
    "zeros": models.baseline.Zeros,
    "single-channel-temporal-regression-12": lambda: models.nn.Temporal(12, 1, 1, models.nn.TemporalRegression(12)),
    "seasonal-temporal-regression": lambda: models.nn.TemporalDate(12, 3, 1, models.nn.SeasonalTemporalRegression(12, 3)),
    "seasonal-temporal-regression-heading": lambda: models.nn.TemporalDate(12, 3, 1, models.nn.SeasonalTemporalRegressionHeading(12, 3)),
    "lygia": lambda: models.nn.TemporalDate(12, 1, 3, models.nn.Lygia(12, 1)),
    "pomponia": lambda: models.nn.TemporalDate(12, 1, 3, models.nn.Pomponia(12, 1)),
    "pomponia-no-mask": lambda: models.nn.TemporalDate(12, 1, 3, models.nn.Pomponia(12, 1, use_mask=False)),
    "pomponia-bias-H-W-L": lambda: models.nn.TemporalDate(12, 1, 3, models.nn.Pomponia(12, 1, use_mask=True, biases="H W L".split())),
    "pomponia-no-mask-bias-H-W-L": lambda: models.nn.TemporalDate(12, 1, 3, models.nn.Pomponia(12, 1, use_mask=False, biases="H W L".split())),
    "pomponia-no-mask-bias-HxL-W-M": lambda: models.nn.TemporalDate(12, 1, 3, models.nn.Pomponia(12, 1, use_mask=False, biases="HxL W M".split())),
}
