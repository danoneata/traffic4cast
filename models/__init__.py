import models.baseline
import models.nn

MODELS = {
    "naive": models.baseline.Naive,
    "zeros": models.baseline.Zeros,
    "temporal-regression-speed-12": lambda: models.nn.Temporal(12, 1, 1, models.nn.TemporalRegression(12)),
}
