import cv2
import numpy as np
import pygame

from audio_manager import AudioManager
from config import (
    CSV_FILENAME,
    MUSIC_SOUND,
    POP_SOUND,
    POSE_MODEL_PATH,
    RADIUS_CLAP,
    RADIUS_TARGET,
    FONT_SIZE,
    FONT_PATH,
    BACKGROUND_IMAGE,
)
from geometry import distance, get_angle
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


def main():
    df, tid, orden, plats, ratt = load_timeline(CSV_FILENAME)
    print(df.head())
    print("start")

    audio = AudioManager(pop_path=POP_SOUND, music_path=MUSIC_SOUND)
    start_time_ms = audio.ensure_music() or pygame.time.get_ticks()
    tracker = PoseTracker(POSE_MODEL_PATH)
    cap = cv2.VideoCapture(0)
    cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("frame", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    renderer = TextRenderer(font_size=FONT_SIZE, font_path=FONT_PATH)
    color1 = (0, 0, 0)
    color2 = (255, 255, 255)
    score = 0
    pointl = []
    indr = np.where(ratt == 1)[0]
    screen_w = None
    screen_h = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            print("No frame from camera, exiting.")
            break
        frame = cv2.flip(frame, 1)

        elapsed_time = (pygame.time.get_ticks() - start_time_ms) / 1000
        dif = tid - elapsed_time
        idd = np.where((dif < 0) & (dif > -1.4))
        idd_text = np.where((dif < 0) & (dif > -1.0))  # clear text after 1s
        dif2 = dif[indr]
        idd2 = indr[np.where((dif2 < 1.5) & (dif2 > -1.5))[0]]
        if len(idd2) > 0:
            print(plats[idd2])

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_timestamp_ms = int(max(0, pygame.time.get_ticks() - start_time_ms))
        pose_landmarks = tracker.detect(frame_rgb, frame_timestamp_ms)

        h, w, _ = frame_rgb.shape
        centers, text_offsets = target_positions(w, h)
        bg_img = None
        try:
            bg_img = cv2.imread(BACKGROUND_IMAGE)
            if bg_img is not None:
                bg_img = cv2.resize(bg_img, (w, h))
        except Exception:
            bg_img = None
        base_frame = bg_img if bg_img is not None else np.full((h, w, 3), 128, dtype=np.uint8)
        frame = draw_targets(base_frame.copy(), centers, alpha=0.7, base_frame=base_frame)

        if len(idd[0]) == 2:
            pr = idd[0][np.where(ratt[idd[0]] == 1)[0]]
            ind2 = np.where(["mitten", "Vhörn", "Hhörn"] == plats[pr])[0]
        else:
            ind2 = []

        frame = draw_text_overlays(frame, idd_text, orden, plats, text_offsets, renderer, color1)

        landmarks = []
        if pose_landmarks:
            for landmark in pose_landmarks:
                landmarks.append((int(landmark.x * w), int(landmark.y * h), (landmark.z * w)))

        if pose_landmarks and len(landmarks) > 20:
            HL = (landmarks[20][0], landmarks[20][1])
            HR = (landmarks[19][0], landmarks[19][1])

            try:
                _ = get_angle((landmarks[12][0], landmarks[12][1]), (landmarks[14][0], landmarks[14][1]), (landmarks[16][0], landmarks[16][1]))
            except Exception:
                pass

            try:
                dis1 = [distance(a, HL) for a in centers]
                dis2 = [distance(a, HR) for a in centers]
                dp1 = np.argmin(dis2)
                dp2 = np.argmin(dis1)
                if dp1 == dp2 and dis1[dp1] < RADIUS_CLAP and dis2[dp2] < RADIUS_CLAP:
                    print("KLAPP")
                    ll = dp1
                    #Clapp circle
                    cv2.circle(frame, centers[dp1], 40, (144, 238, 144), -1)
                    if len(idd2) > 0 and ["mitten", "Vhörn", "Hhörn"][ll] == plats[idd2]:
                        pr2 = idd2
                        if pr2 not in pointl:
                            score += 1
                            pointl.append(pr2)
                    if False and len(ind2) > 0:
                        if ll == ind2:
                            score += 1
                            print("SCORE!")
            except Exception:
                pass
            tracker.draw(frame, pose_landmarks)
            cv2.circle(frame, HR, 40, (0, 255, 0), 2)
            cv2.circle(frame, HL, 40, (0, 0, 255), 2)

        cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
        #cv2.setWindowProperty("frame", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        ts = "Score: " + str(score)
        text_w, text_h = renderer.size(ts)
        textX = (w - text_w) / 2
        textY = (h + text_h) / 2 + int(h / 3.5)
        frame = renderer.draw(frame, [{"text": ts, "x": int(textX), "y": int(textY), "color": color2}])
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
