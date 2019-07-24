from models.baseline import Naive, Zeros, TemporalRegression

MODELS = {
    "naive": Naive,
    "zeros": Zeros,
    "temporal-regression-speed-12": lambda: TemporalRegression("speed", 12),
}
