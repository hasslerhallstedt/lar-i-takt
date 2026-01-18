import os

# Assets and data
# CSV_FILENAME = "resources/nyfil_gang-1.csv"
CSV_FILENAME = "resources/lar-i-takt-content.json"
POP_SOUND = "resources/pop.wav"
MUSIC_SOUND = "resources/Back on track.wav"
POSE_MODEL_PATH = os.getenv("POSE_LANDMARKER_MODEL_PATH", "pose_landmarker_full.task")
FONT_SIZE = 96
FONT_PATH = "/Users/maan/src/maan/lar-i-takt/resources/Roboto-Black.ttf"
BACKGROUND_IMAGE = "resources/background.jpg"
#FONT_PATH = "/System/Library/Fonts/Supplemental/Arial.ttf"  # change if you want another TTF

# Visuals and thresholds
RADIUS_TARGET = int(1.5 * 70)
RADIUS_CLAP = int(3 * 70)
