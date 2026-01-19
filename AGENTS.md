# AGENTS.md

## Project summary
- Pose-based clapping/timing game built with MediaPipe Tasks, OpenCV, and Pygame.
- Uses webcam input, renders targets/text overlays, and scores claps based on hand positions and timing windows.
- Timeline can be loaded from CSV or JSON (schema in `resources/lar-i-takt-schema.json`).

## Key files
- `po.py`: main game loop; orchestrates timeline load, audio, pose detection, rendering, and scoring.
- `config.py`: constants for asset paths and thresholds.
- `timeline.py`: loads timeline from CSV (`tid;ord;plats;ratt`) or JSON per `resources/lar-i-takt-schema.json`.
- `pose_tracker.py`: wraps MediaPipe Tasks PoseLandmarker model.
- `face_mesh_tracker.py`: optional MediaPipe Tasks face mesh setup/detect/draw when enabled.
- `audio_manager.py`: handles music/pop sounds.
- `geometry.py`: small math helpers.
- Assets: `nyfil_gang-1.csv`, `pop.wav`, `Back on track.wav`, `pose_landmarker_full.task` (model), optional JSON content (`resources/lar-i-takt-content.json`).

## Runtime requirements
- Python 3 with: `opencv-python`, `mediapipe`, `numpy`, `pandas`, `pygame`, `matplotlib`.
- A working webcam.
- Audio files in the working directory: `pop.wav` and any other `.wav` referenced in `po.py` (e.g. `Back on track.wav`).
- Optional: face mesh model `face_landmarker.task` if `SHOW_FACE_MESH` is enabled.

## How to run
- Optional (macOS): create and activate a virtual environment first:
  - `python3 -m venv .venv`
  - `source .venv/bin/activate`
- Install dependencies:
  - `pip install -r requirements.txt`
- Run from the project root so relative paths resolve:
  - `python3 po.py`
- The app opens to a start screen; click "Start" to trigger a 5-second countdown before gameplay begins.
- Quit the app by pressing `q` in the OpenCV window.

## Architecture & APIs
- MediaPipe Tasks PoseLandmarker (`mediapipe.tasks.python.vision.PoseLandmarker`) with `detect_for_video` on `ImageFormat.SRGB` frames; model path from `POSE_LANDMARKER_MODEL_PATH` or `pose_landmarker_full.task`.
- Optional MediaPipe Tasks FaceLandmarker when `SHOW_FACE_MESH` is true; model path from `FACE_LANDMARKER_MODEL_PATH` or `face_landmarker.task`.
- OpenCV for capture/render and simple overlays; Pygame for audio and timing.
- Modular layout: `po.py` (orchestrator), `config.py` (paths/constants), `timeline.py` (CSV/JSON ingestion), `pose_tracker.py` (pose model setup/detect/draw), `face_mesh_tracker.py` (face mesh setup/detect/draw), `audio_manager.py`, `geometry.py`.
- Targets are fixed to three positions; clap detection compares both hands to target centers within `RADIUS_CLAP`.

## Data format notes
- CSV: `;` separator, UTF-8. Columns: `tid` (float seconds), `ord` (text), `plats` (`mitten|Vhörn|Hhörn`), `ratt` (1=scoreable).
- JSON: see `resources/lar-i-takt-schema.json`; example in `resources/lar-i-takt-content.json`. `timestamp` + `question` + optional responses with `correct="true"` map into the same internal columns, and response-level timestamps are used when present.

## Agent guidelines
- Preserve current file naming and relative paths; the game expects assets in the repo root unless paths are updated in `config.py`.
- Avoid large refactors unless requested; modular structure is already in place.
- If adding new assets, keep them in the project root and update `config.py` accordingly.
