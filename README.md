# lar-i-takt

Prototype pose-based clapping/timing game using MediaPipe Tasks + OpenCV + Pygame. A webcam tracks your hands, shows targets/text overlays from a CSV or JSON timeline, and scores claps when you hit the right spot at the right time. Audio cues come from the bundled WAV files.

## Quick start
```bash
python3 -m venv .venv && source .venv/bin/activate   # optional
pip install -r requirements.txt
curl -L -o pose_landmarker_full.task https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task
# Optional: face mesh overlay
curl -L -o face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
python3 po.py
```

Requirements:
- Python 3, webcam, audio output
- Packages: `mediapipe>=0.10.20`, `opencv-python`, `numpy`, `pandas`, `pygame`, `matplotlib`
- Assets in repo root: `pop.wav`, `Back on track.wav`, timeline (`nyfil_gang-1.csv` or `resources/lar-i-takt-content.json`), pose model `pose_landmarker_full.task`
- Optional assets: face mesh model `face_landmarker.task` (enable via `SHOW_FACE_MESH = True` in `config.py`)

## How it works
- `po.py` orchestrates the loop; timelines are loaded via `timeline.py`:
  - CSV: `tid;ord;plats;ratt` (UTF-8, semicolon-separated).
  - JSON: conforms to `resources/lar-i-takt-schema.json` (see `resources/lar-i-takt-content.json`); uses question timestamps and response-level timestamps when provided.
- MediaPipe Tasks PoseLandmarker runs per frame (`pose_landmarker_full.task` or `POSE_LANDMARKER_MODEL_PATH` env var).
- Optional: MediaPipe Tasks FaceLandmarker renders a face mesh overlay when `SHOW_FACE_MESH` is enabled and `face_landmarker.task` (or `FACE_LANDMARKER_MODEL_PATH`) is available.
- Three targets (left, middle, right). Claps are detected when both hands enter the same target radius near scheduled times (`ratt=1` rows).
- Press `q` to quit.

## Architecture
- Modules: `po.py` (orchestration), `config.py` (paths/constants), `timeline.py` (CSV/JSON ingest), `pose_tracker.py` (pose Tasks setup/detect/draw), `face_mesh_tracker.py` (face mesh Tasks setup/detect/draw), `audio_manager.py`, `geometry.py`.
- MediaPipe API: `vision.PoseLandmarker.detect_for_video` and (optionally) `vision.FaceLandmarker.detect_for_video` on `ImageFormat.SRGB` frames; landmark drawing uses `mediapipe.framework.formats.landmark_pb2` and available connection maps.
- OpenCV handles capture and overlays; Pygame handles audio and timing. Targets are fixed to three positions; clap is scored when both hands enter the same target within `RADIUS_CLAP`.

## Notes
- Run from the repo root so relative paths to audio/CSV/model resolve.
- If you see “No frame from camera”, check webcam permissions/device availability.
- Missing model file raises a clear error; download it as shown above or point `POSE_LANDMARKER_MODEL_PATH` to your copy.
