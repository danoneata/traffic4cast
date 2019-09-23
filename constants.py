import os

ROOT = os.environ.get("ROOT", "data")
CHANNELS = ["volume", "speed", "heading"]
CITIES = ["Berlin", "Istanbul", "Moscow"]

BERLIN_START_FRAMES = [30, 69, 126, 186, 234]
ISTANBUL_START_FRAMES = [57, 114, 174, 222, 258]
MOSCOW_START_FRAMES = ISTANBUL_START_FRAMES

N_FRAMES = 3  # Predict this many frames into the future
SUBMISSION_FRAMES = {
    city: [s + i for s in start_frames for i in range(N_FRAMES)]
    for start_frames, city in
    zip([BERLIN_START_FRAMES, ISTANBUL_START_FRAMES, MOSCOW_START_FRAMES], CITIES)
}

EVALUATION_SHAPE = (5 * N_FRAMES, 495, 436, 3)
SUBMISSION_SHAPE = (5, N_FRAMES, 495, 436, 3)
