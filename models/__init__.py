from models.baseline import Naive, Zeros, TemporalRegression
import models.nn

MODELS = {
    "naive": Naive,
    "zeros": Zeros,
    "generic-temporal": lambda: models.nn.Temporal(12, 1, ["Speed"], models.nn.TemporalRegression(12)),
}
