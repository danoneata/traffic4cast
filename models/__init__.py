from models.baseline import Naive, Zeros, TemporalRegression

MODELS = {
    "naive": Naive,
    "zeros": Zeros,
    "temp-regr": TemporalRegression,
}
