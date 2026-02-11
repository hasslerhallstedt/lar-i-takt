import os

# Assets and data
#CSV_FILENAME = "resources/nyfil_gang-1.csv"
CSV_FILENAME = "resources/Content_Avicii.json"
POP_SOUND = "resources/pop.wav"
MUSIC_SOUND = "resources/Avicii.mp3"
POSE_MODEL_PATH = os.getenv("POSE_LANDMARKER_MODEL_PATH", "pose_landmarker_full.task")
FACE_MODEL_PATH = os.getenv("FACE_LANDMARKER_MODEL_PATH", "face_landmarker.task")
FONT_SIZE = 96
FONT_PATH = "resources/Roboto-Black.ttf"
BACKGROUND_IMAGE = "resources/background.jpg"
SHOW_FACE_MESH = True
#FONT_PATH = "/System/Library/Fonts/Supplemental/Arial.ttf"  # change if you want another TTF

# Visuals and thresholds
RADIUS_TARGET = int(180)
RADIUS_CLAP = int(2 * RADIUS_TARGET )
