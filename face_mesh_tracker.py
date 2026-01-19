import os

from mediapipe import Image, ImageFormat
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

try:
    from mediapipe.framework.formats import landmark_pb2
    from mediapipe.python.solutions import drawing_utils as mp_drawing
    from mediapipe.python.solutions import face_mesh as mp_face_mesh_solutions
except Exception:
    landmark_pb2 = None
    mp_drawing = None
    mp_face_mesh_solutions = None


class FaceMeshTracker:
    def __init__(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Face landmarker model not found at '{model_path}'. "
                "Set FACE_LANDMARKER_MODEL_PATH or place the .task file in the project root."
            )
        self.connections = (
            mp_face_mesh_solutions.FACEMESH_TESSELATION if mp_face_mesh_solutions else None
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=model_path),
                running_mode=vision.RunningMode.VIDEO,
                num_faces=1,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
            )
        )

    def detect(self, frame_rgb, timestamp_ms):
        mp_image = Image(image_format=ImageFormat.SRGB, data=frame_rgb)
        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
        if result and result.face_landmarks:
            return result.face_landmarks[0]
        return None

    def draw(self, frame, face_landmarks):
        if not face_landmarks or not (landmark_pb2 and mp_drawing and self.connections):
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
                for landmark in face_landmarks
            ]
        )
        drawing_spec = mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=1, circle_radius=1)
        connection_spec = mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=1, circle_radius=1)
        mp_drawing.draw_landmarks(
            frame,
            landmarks_proto,
            self.connections,
            landmark_drawing_spec=drawing_spec,
            connection_drawing_spec=connection_spec,
        )
