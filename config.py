import os

# Assets and data
CSV_FILENAME = "resources/nyfil_gang-1.csv"
POP_SOUND = "resources/pop.wav"
MUSIC_SOUND = "resources/Back on track.wav"
POSE_MODEL_PATH = os.getenv("POSE_LANDMARKER_MODEL_PATH", "pose_landmarker_full.task")

# Visuals and thresholds
TEXT_SCALE = 1.5 * 2
RADIUS_TARGET = int(1.5 * 70)
RADIUS_CLAP = int(3 * 70)
