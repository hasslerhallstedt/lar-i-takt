# AGENTS.md

## Project summary
- Single-file Python prototype that uses MediaPipe + OpenCV + Pygame to run a clapping/pose timing game driven by a CSV timeline.
- Uses webcam input, renders targets/text overlays, and scores claps based on hand positions and timing windows.

## Key files
- `po.py`: main script; loads CSV timeline, initializes audio, processes webcam frames, and scores claps.
- `nyfil_gang-1.csv`: timeline data with columns `tid`, `ord`, `plats`, `ratt` (semicolon-separated).
- `pop.wav`: sound effect referenced by `po.py`.

## Runtime requirements
- Python 3 with: `opencv-python`, `mediapipe`, `numpy`, `pandas`, `pygame`, `matplotlib`.
- A working webcam.
- Audio files in the working directory: `pop.wav` and any other `.wav` referenced in `po.py` (e.g. `Back on track.wav`).

## How to run
- Optional (macOS): create and activate a virtual environment first:
  - `python3 -m venv .venv`
  - `source .venv/bin/activate`
- Install dependencies:
  - `pip install -r requirements.txt`
- Run from the project root so relative paths resolve:
  - `python3 po.py`
- Quit the app by pressing `q` in the OpenCV window.

## Data format notes
- `nyfil_gang-1.csv` uses `;` as the separator.
- CSV is read as UTF-8 (`encoding="utf-8"` in `po.py`).
- `plats` values expected by the script are `mitten`, `Vhörn`, `Hhörn`.
- `ratt` is used to flag scoring moments (1 = scoreable).

## Agent guidelines
- Preserve the current file naming and relative paths; the script relies on them.
- Avoid large refactors unless requested; this is a single-file prototype.
- If adding new assets, keep them in the project root and update `po.py` accordingly.
