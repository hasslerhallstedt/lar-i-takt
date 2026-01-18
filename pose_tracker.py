from mediapipe import Image, ImageFormat
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import os

try:
    from mediapipe.framework.formats import landmark_pb2
    from mediapipe.python.solutions import drawing_utils as mp_drawing
    from mediapipe.python.solutions import pose as mp_pose_solutions
except Exception:
    landmark_pb2 = None
    mp_drawing = None
    mp_pose_solutions = None


class PoseTracker:
    def __init__(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Pose landmarker model not found at '{model_path}'. "
                "Set POSE_LANDMARKER_MODEL_PATH or place the .task file in the project root."
            )
        self.connections = mp_pose_solutions.POSE_CONNECTIONS if mp_pose_solutions else None
        self.landmarker = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=model_path),
                running_mode=vision.RunningMode.VIDEO,
                num_poses=1,
                output_segmentation_masks=False,
            )
        )

    def detect(self, frame_rgb, timestamp_ms):
        mp_image = Image(image_format=ImageFormat.SRGB, data=frame_rgb)
        pose_results = self.landmarker.detect_for_video(mp_image, timestamp_ms)
        return pose_results.pose_landmarks[0] if pose_results and pose_results.pose_landmarks else None

    def draw(self, frame, pose_landmarks):
        if not pose_landmarks or not (landmark_pb2 and mp_drawing and self.connections):
            return
        landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        landmarks_proto.landmark.extend(
            [
                landmark_pb2.NormalizedLandmark(
                    x=landmark.x,
                    y=landmark.y,
                    z=landmark.z,
                    visibility=landmark.visibility,
                )
                for landmark in pose_landmarks
            ]
        )
        mp_drawing.draw_landmarks(frame, landmarks_proto, self.connections)
