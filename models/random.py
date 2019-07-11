import numpy as np

class Random:
    def __init__(self, city):
        pass

    def predict(self, date, n_frames):
        # TODO Are all the images the same size?
        H, W, C = 495, 436, 3
        return np.random.randn(n_frames, H, W, C)
