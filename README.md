# lar-i-takt

## System Overview
`lar-i-takt` is a single-file Python prototype that turns a CSV-driven timeline into a timing game.
It uses MediaPipe + OpenCV to track hands from a webcam, Pygame to play audio, and draws targets
and text overlays on the video feed. Claps are scored when hands meet within the timing window
and at the target position specified by the CSV.

## How It Works
- A CSV timeline (`nyfil_gang-1.csv`) defines when targets appear, what text to show, and where
  the target is placed (`mitten`, `Vhörn`, `Hhörn`).
- `po.py` loads the timeline, plays audio cues, reads webcam frames, and uses MediaPipe landmarks
  to infer hand positions and clap timing.
- Each frame renders targets and status text in OpenCV, while scoring logic tracks "on time" claps
  based on the `ratt` flag and timing windows.
- A pop sound (`pop.wav`) plays on scoring events.

## Key Files
- `po.py`: main script (timeline loading, webcam loop, rendering, scoring).
- `nyfil_gang-1.csv`: timeline data (`tid`, `ord`, `plats`, `ratt`, semicolon-separated).
- `pop.wav`: clap/score sound effect.
- `Back on track.wav`: background audio referenced by the script.

## Runtime Requirements
- Python 3 with: `opencv-python`, `mediapipe`, `numpy`, `pandas`, `pygame`, `matplotlib`.
- A working webcam.
- Audio files in the project root that are referenced by `po.py`.

## Launch
1) Create and activate a virtual environment (optional):
   - `python3 -m venv .venv`
   - `source .venv/bin/activate`
2) Install dependencies:
   - `pip install -r requirements.txt`
3) Run from the project root so relative paths resolve:
   - `python3 po.py`
4) Quit the app by pressing `q` in the OpenCV window.
