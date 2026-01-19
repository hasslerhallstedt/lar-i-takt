import cv2
import numpy as np
import pygame
from dataclasses import dataclass

from audio_manager import AudioManager
from config import (
    CSV_FILENAME,
    MUSIC_SOUND,
    POP_SOUND,
    FACE_MODEL_PATH,
    POSE_MODEL_PATH,
    RADIUS_CLAP,
    RADIUS_TARGET,
    FONT_SIZE,
    FONT_PATH,
    BACKGROUND_IMAGE,
    SHOW_FACE_MESH,
)
from geometry import distance
from face_mesh_tracker import FaceMeshTracker
from pose_tracker import PoseTracker
from timeline import load_timeline
from text_renderer import TextRenderer


def target_positions(width, height):
    center_left = (int(width / 5), int(height / 4))
    center_mid = (int(width / 2), int(height / 2))
    center_right = (int(width - width / 5), int(height / 4))

    offset_x = int(width / 2 - width / 5)
    offset_y = int(height / 2 - height / 4)+70 #Y offset for answers
    text_offsets = {
        "mitten": (0, -70),
        "Vhörn": (-offset_x, -offset_y),
        "Hhörn": (offset_x, -offset_y),
    }
    return (center_left, center_mid, center_right), text_offsets


def draw_targets(frame, centers, alpha, base_frame):
    cv2.circle(frame, centers[1], RADIUS_TARGET, (255, 255, 255), -1)
    cv2.circle(frame, centers[0], RADIUS_TARGET, (255, 255, 255), -1)
    cv2.circle(frame, centers[2], RADIUS_TARGET, (255, 255, 255), -1)
    return cv2.addWeighted(frame, alpha, base_frame, 1 - alpha, 0)


def draw_text_overlays(frame, idd, orden, plats, text_offsets, renderer, color):
    h, w, _ = frame.shape
    items = []
    for idx in idd[0]:
        word = orden[idx]
        place = plats[idx]
        if len(str(place)) < 4:
            place = "mitten"
        try:
            text_w, text_h = renderer.size(str(word))
            text_x = (w - text_w) / 2 + text_offsets[place][0]
            text_y = (h + text_h) / 2 + text_offsets[place][1]
            items.append({"text": word, "x": int(text_x), "y": int(text_y), "color": color})
        except ValueError:
            pass
    if items:
        frame = renderer.draw(frame, items)
    return frame

def draw_face_landmarks_points(image_bgr, face_landmarks, radius=1, thickness=-2):
    """Draw points for one face. face_landmarks: List[NormalizedLandmark]."""
    h, w = image_bgr.shape[:2]
    for lm in face_landmarks:
        x = int(lm.x * w)
        y = int(lm.y * h)
        cv2.circle(image_bgr, (x, y), radius, (64, 64, 64), thickness, cv2.LINE_AA)
# Pose landmark indices
LS, RS = 11, 12
LE, RE = 13, 14
LW, RW = 15, 16

ARM_EDGES = [
    (LS, LE), (LE, LW),   # left arm
    (RS, RE), (RE, RW),   # right arm
]

def draw_arms_bgr(image_bgr, pose_landmarks, thickness=4):
    """
    image_bgr: OpenCV image (H,W,3) BGR
    pose_landmarks: a list of NormalizedLandmark (result.pose_landmarks[0])
    """
    h, w = image_bgr.shape[:2]

    def to_px(lm):
        return int(lm.x * w), int(lm.y * h)

    for a, b in ARM_EDGES:
        pa = to_px(pose_landmarks[a])
        pb = to_px(pose_landmarks[b])
        cv2.line(image_bgr, pa, pb, (64, 64, 64), thickness, cv2.LINE_AA)

    return image_bgr

def draw_hand_circles(frame, hand_right, hand_left, radius=40, color=(64, 64, 64), thickness=4):
    """Draw circles around detected hand positions.

    Args:
        frame: The image frame to draw on
        hand_right: Tuple (x, y) for right hand position
        hand_left: Tuple (x, y) for left hand position
        radius: Circle radius in pixels
        color: BGR color tuple
        thickness: Line thickness (1 for outline, -1 for filled)
    """
    cv2.circle(frame, hand_right, radius, color, thickness)
    cv2.circle(frame, hand_left, radius, color, thickness)
    return frame


@dataclass
class StartState:
    started: bool = False
    triggered: bool = False
    countdown_start_ms: int | None = None
    start_time_ms: int | None = None
    countdown_seconds: int = 5


def handle_start_screen(frame, renderer, color, start_state, click_state, audio):
    """Render start button + countdown; update state when clicked/finished."""
    h, w, _ = frame.shape
    btn_w, btn_h = 240, 100
    btn_x1 = int((w - btn_w) / 2)
    btn_y1 = int((h - btn_h) / 2)
    btn_x2 = btn_x1 + btn_w
    btn_y2 = btn_y1 + btn_h

    cv2.rectangle(frame, (btn_x1, btn_y1), (btn_x2, btn_y2), (50, 50, 50), -1)
    cv2.rectangle(frame, (btn_x1, btn_y1), (btn_x2, btn_y2), (255, 255, 255), 2)
    start_text = "Start"
    st_w, st_h = renderer.size(start_text)
    frame = renderer.draw(
        frame,
        [
            {
                "text": start_text,
                "x": btn_x1 + int((btn_w - st_w) / 2),
                "y": btn_y1 + int((btn_h - st_h) / 2),
                "color": color,
            }
        ],
    )

    if click_state.get("pos"):
        click_x, click_y = click_state["pos"]
        click_state["pos"] = None
        if btn_x1 <= click_x <= btn_x2 and btn_y1 <= click_y <= btn_y2:
            start_state.triggered = True
            start_state.countdown_start_ms = pygame.time.get_ticks()

    if start_state.triggered and start_state.countdown_start_ms is not None:
        elapsed = (pygame.time.get_ticks() - start_state.countdown_start_ms) / 1000
        remaining = start_state.countdown_seconds - int(elapsed)
        if remaining <= 0:
            start_state.started = True
            start_state.start_time_ms = audio.ensure_music() or pygame.time.get_ticks()
        else:
            countdown_text = f"Starting in {remaining}"
            c_w, c_h = renderer.size(countdown_text)
            frame = renderer.draw(
                frame,
                [
                    {
                        "text": countdown_text,
                        "x": int((w - c_w) / 2),
                        "y": int(btn_y1 - c_h - 10),
                        "color": color,
                    }
                ],
            )
    else:
        prompt = "Click Start to begin"
        p_w, p_h = renderer.size(prompt)
        frame = renderer.draw(
            frame,
            [
                {
                    "text": prompt,
                    "x": int((w - p_w) / 2),
                    "y": int(btn_y2 + p_h),
                    "color": color,
                }
            ],
        )

    return frame, start_state


def compute_timeline_windows(tid, start_time_ms, indr):
    elapsed_time = (pygame.time.get_ticks() - start_time_ms) / 1000 if start_time_ms else 0
    dif = tid - elapsed_time
    idd = np.where((dif < 0) & (dif > -1.4))
    idd_text = np.where((dif < 0) & (dif > -1.0))  # clear text after 1s
    dif2 = dif[indr]
    idd2 = indr[np.where((dif2 < 1.5) & (dif2 > -1.5))[0]]
    return elapsed_time, idd, idd_text, idd2


def detect_landmarks(tracker, face_tracker, frame_rgb, timestamp_ms):
    pose_landmarks = tracker.detect(frame_rgb, timestamp_ms)
    face_landmarks = face_tracker.detect(frame_rgb, timestamp_ms) if face_tracker else None
    return pose_landmarks, face_landmarks


def draw_score(frame, renderer, color, score):
    h, w, _ = frame.shape
    ts = f"Score: {score}"
    text_w, text_h = renderer.size(ts)
    textX = (w - text_w) / 2
    textY = (h + text_h) / 2 + int(h / 3.5)
    return renderer.draw(frame, [{"text": ts, "x": int(textX), "y": int(textY), "color": color}])


def process_pose(frame, pose_landmarks, centers, plats, idd2, score, pointl, tracker):
    if not pose_landmarks or len(pose_landmarks) <= 20:
        return frame, score, pointl

    h, w, _ = frame.shape
    landmarks = [
        (int(landmark.x * w), int(landmark.y * h), (landmark.z * w)) for landmark in pose_landmarks
    ]
    HL = (landmarks[20][0], landmarks[20][1])
    HR = (landmarks[19][0], landmarks[19][1])
    SL = (landmarks[12][0], landmarks[12][1])
    SR = (landmarks[11][0], landmarks[11][1])

    try:
        dis1 = [distance(a, HL) for a in centers]
        dis2 = [distance(a, HR) for a in centers]
        dp1 = np.argmin(dis2)
        dp2 = np.argmin(dis1)
        if dp1 == dp2 and dis1[dp1] < RADIUS_CLAP and dis2[dp2] < RADIUS_CLAP:
            ll = dp1
            cv2.circle(frame, centers[dp1], 40, (144, 238, 144), -1)
            if len(idd2) > 0:
                hit_idx = int(idd2[0])
                if ["mitten", "Vhörn", "Hhörn"][ll] == plats[hit_idx]:
                    if hit_idx not in pointl:
                        score += 1
                        pointl.append(hit_idx)
    except Exception:
        pass

    tracker.draw(frame, pose_landmarks)
    draw_hand_circles(frame, HL, HR, radius=80, thickness=4)
    cv2.line(frame, SL, HL, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.line(frame, SR, HR, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.line(frame, SR, SL, (0, 255, 0), 2, cv2.LINE_AA)
    return frame, score, pointl


def draw_face_overlays(frame, face_landmarks, face_tracker, pose_landmarks):
    if not face_landmarks:
        return frame
    face_tracker.draw(frame, face_landmarks)
    draw_face_landmarks_points(frame, face_landmarks)
    if pose_landmarks:
        draw_arms_bgr(frame, pose_landmarks)
    return frame


def main():
    df, tid, orden, plats, ratt = load_timeline(CSV_FILENAME)
    print(df.head())
    print("start")

    audio = AudioManager(pop_path=POP_SOUND, music_path=MUSIC_SOUND)
    tracker = PoseTracker(POSE_MODEL_PATH)
    face_tracker = None
    if SHOW_FACE_MESH:
        try:
            face_tracker = FaceMeshTracker(FACE_MODEL_PATH)
        except FileNotFoundError as exc:
            print(exc)
            face_tracker = None
    cap = cv2.VideoCapture(0)
    cv2.namedWindow("frame", cv2.WINDOW_NORMAL)

    renderer = TextRenderer(font_size=FONT_SIZE, font_path=FONT_PATH)
    color1 = (0, 0, 0)
    color2 = (255, 255, 255)
    score = 0
    pointl = []
    indr = np.where(ratt == 1)[0]
    screen_w = None
    screen_h = None
    centers = None
    text_offsets = None
    start_state = StartState()
    click_state = {"pos": None}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            param["pos"] = (x, y)

    cv2.setMouseCallback("frame", on_mouse, click_state)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            print("No frame from camera, exiting.")
            break
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        if centers is None or text_offsets is None:
            centers, text_offsets = target_positions(w, h)

        frame_base = draw_targets(frame.copy(), centers, alpha=0.7, base_frame=frame)

        if not start_state.started:
            frame_display, start_state = handle_start_screen(
                frame_base.copy(), renderer, color2, start_state, click_state, audio
            )
            if screen_w is None or screen_h is None:
                try:
                    wx, wy, ww, wh = cv2.getWindowImageRect("frame")
                    if ww > 0 and wh > 0:
                        screen_w, screen_h = ww, wh
                except Exception:
                    screen_w, screen_h = w, h
            output_frame = cv2.resize(frame_display, (screen_w or w, screen_h or h))
            cv2.imshow("frame", output_frame)
            if cv2.waitKey(1) == ord("q"):
                break
            continue

        _, idd, idd_text, idd2 = compute_timeline_windows(tid, start_state.start_time_ms, indr)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_timestamp_ms = int(max(0, pygame.time.get_ticks() - (start_state.start_time_ms or 0)))
        pose_landmarks, face_landmarks = detect_landmarks(
            tracker, face_tracker, frame_rgb, frame_timestamp_ms
        )

        frame = draw_text_overlays(frame_base.copy(), idd_text, orden, plats, text_offsets, renderer, color1)
        frame, score, pointl = process_pose(frame, pose_landmarks, centers, plats, idd2, score, pointl, tracker)
        frame = draw_face_overlays(frame, face_landmarks, face_tracker, pose_landmarks)
        frame = draw_score(frame, renderer, color2, score)

        if screen_w is None or screen_h is None:
            try:
                wx, wy, ww, wh = cv2.getWindowImageRect("frame")
                if ww > 0 and wh > 0:
                    screen_w, screen_h = ww, wh
            except Exception:
                screen_w, screen_h = w, h
        output_frame = cv2.resize(frame, (screen_w or w, screen_h or h))
        cv2.imshow("frame", output_frame)

        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
