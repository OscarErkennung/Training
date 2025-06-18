import cv2
import os
import sys
from train_mobilenet import IMG_SIZE


def extract_frames(video_path: str, output_dir: str, interval_sec=0.5):
    # Create output folder if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("cannot open video file")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    interval_frames = int(fps * interval_sec)
    saved_count = 0

    print(
        f"extracting 1 frame every {interval_sec} seconds "
        f"(~{interval_frames} frames)...")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps
    current_time = 0.0
    saved_count = 0

    while current_time < duration_sec:
        # jump to time in milliseconds
        cap.set(cv2.CAP_PROP_POS_MSEC, current_time * 1000)
        ret, frame = cap.read()
        if not ret:
            break

        resized = cv2.resize(frame, IMG_SIZE)
        filename = os.path.join(
            output_dir, f"{video_path.split('/')[-1][:-4]}_{saved_count:05d}.jpg")
        cv2.imwrite(filename, resized)
        saved_count += 1

        current_time += interval_sec

    cap.release()
    print(f"saved {saved_count} frames to `{output_dir}`")


args = sys.argv[1:]
args = dict(tuple(arg.split("=")) for arg in args)

interval_sec = float(args.get("--iv", 0.5))

for root, _, files in os.walk("video-data"):
    for file in files:
        extract_frames(
            video_path=os.path.join(root,  file),
            output_dir=os.path.join("data1", root.split("/")[1]),
            interval_sec=interval_sec
        )
