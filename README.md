# lar-i-takt

Prototype pose-based clapping/timing game using MediaPipe Tasks + OpenCV + Pygame. A webcam tracks your hands, shows targets/text overlays from a CSV timeline, and scores claps when you hit the right spot at the right time. Audio cues come from the bundled WAV files.

## Quick start
```bash
python3 -m venv .venv && source .venv/bin/activate   # optional
pip install -r requirements.txt
curl -L -o pose_landmarker_full.task https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task
python3 po.py
```

Requirements:
- Python 3, webcam, audio output
- Packages: `mediapipe>=0.10.20`, `opencv-python`, `numpy`, `pandas`, `pygame`, `matplotlib`
- Assets in repo root: `pop.wav`, `Back on track.wav`, CSV timeline `nyfil_gang-1.csv`

## How it works
- `po.py` loads the CSV (`tid;ord;plats;ratt`) and plays background audio.
- MediaPipe Tasks PoseLandmarker runs per frame (`pose_landmarker_full.task` or `POSE_LANDMARKER_MODEL_PATH` env var).
- Three targets (left, middle, right). Claps are detected when both hands enter the same target radius near scheduled times (`ratt=1` rows).
- Press `q` to quit.

## Notes
- Run from the repo root so relative paths to audio/CSV/model resolve.
- If you see “No frame from camera”, check webcam permissions/device availability.
- Missing model file raises a clear error; download it as shown above or point `POSE_LANDMARKER_MODEL_PATH` to your copy.
