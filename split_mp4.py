import cv2
import os
import sys

IMG_SIZE = (224, 224)
TRAIN_VAL_SPLIT = 0.2


def video_duration_sec(video_path: str):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return total_frames / cap.get(cv2.CAP_PROP_FPS)


args = sys.argv[1:]
args = dict(tuple(arg.split("=", 1)) for arg in args)

path = args.get("--path")
num_images = int(args.get("--n"))

total_duration = 0.0
for root, _, files in os.walk(path):
    for file in files:
        total_duration += video_duration_sec(os.path.join(root, file))

interval_sec = total_duration / num_images
print(f"extracting 1 frame every {interval_sec:.4f} seconds")

train_val_cutoff = int(num_images * (1 - TRAIN_VAL_SPLIT))
print(
    f"creating {train_val_cutoff} images for training and "
    f"~{num_images - train_val_cutoff} images for validation"
)

train_output_dir = os.path.join("data/train", path.split("/", 1)[1])
val_output_dir = os.path.join("data/val", path.split("/", 1)[1])

os.makedirs("data", exist_ok=True)
os.makedirs(train_output_dir, exist_ok=True)
os.makedirs(val_output_dir, exist_ok=True)

total_saved = 0

for root, _, files in os.walk(path):
    for file in files:
        video_path = os.path.join(root, file)
        # Create output folder if it doesn't exist

        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise LookupError("cannot open video file")

        print(f"processing {video_path}")

        saved_count = 0
        duration_sec = video_duration_sec(video_path)
        current_time = 0.0

        while current_time < duration_sec:
            # jump to time in milliseconds
            cap.set(cv2.CAP_PROP_POS_MSEC, current_time * 1000)
            ret, frame = cap.read()
            if not ret:
                break

            resized = cv2.resize(frame, IMG_SIZE)
            filename = os.path.join(
                train_output_dir if total_saved < train_val_cutoff else val_output_dir,
                f"{video_path.split('/')[-1][:-4]}_{saved_count:05d}.jpg",
            )
            cv2.imwrite(filename, resized)
            saved_count += 1
            total_saved += 1

            current_time += interval_sec

        cap.release()
