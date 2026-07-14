"""
Extract face-cropped frame sequences from Celeb-DF-v2 videos.

Expected input structure (standard Celeb-DF-v2 unzip):
    celeb_df/
        Celeb-real/*.mp4        (real videos)
        Celeb-synthesis/*.mp4   (fake videos)
        YouTube-real/*.mp4      (additional real videos, optional)
        List_of_testing_videos.txt   (optional — official test split)

Output structure (frame sequences, grouped by video, split by VIDEO not frame
to avoid data leakage between train/valid/test):
    frames/
        train/real/<video_id>/frame_000.jpg ... frame_015.jpg
        train/fake/<video_id>/frame_000.jpg ...
        valid/real/<video_id>/...
        valid/fake/<video_id>/...
        test/real/<video_id>/...
        test/fake/<video_id>/...

Usage:
    python3 extract_frames.py --celeb_dir celeb_df --output_dir frames --frames_per_video 16
"""
import argparse
import os
import random
import cv2
from facenet_pytorch import MTCNN
import torch


def get_video_list(celeb_dir):
    """Returns list of (video_path, label) where label 0=real, 1=fake."""
    videos = []
    real_dirs = ["Celeb-real", "YouTube-real"]
    fake_dirs = ["Celeb-synthesis"]

    for d in real_dirs:
        full = os.path.join(celeb_dir, d)
        if os.path.isdir(full):
            for f in os.listdir(full):
                if f.lower().endswith(".mp4"):
                    videos.append((os.path.join(full, f), 0))

    for d in fake_dirs:
        full = os.path.join(celeb_dir, d)
        if os.path.isdir(full):
            for f in os.listdir(full):
                if f.lower().endswith(".mp4"):
                    videos.append((os.path.join(full, f), 1))

    return videos


def extract_faces_from_video(video_path, mtcnn, num_frames=16, img_size=224):
    """Reads a video, samples num_frames evenly spaced frames, detects+crops
    the largest face in each. Returns list of PIL-compatible face crops
    (as numpy arrays), or None if face detection fails on too many frames."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames < num_frames:
        cap.release()
        return None  # video too short, skip

    frame_indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
    faces = []

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # MTCNN expects RGB, returns cropped+aligned face tensor directly if given img_size
        try:
            face_crop = mtcnn(frame_rgb)
        except Exception:
            face_crop = None

        if face_crop is None:
            continue  # no face detected in this frame, skip it

        # face_crop is a tensor [3, H, W] in [0,1] range from facenet_pytorch's default;
        # convert back to a standard uint8 numpy image for saving
        face_np = (face_crop.permute(1, 2, 0).numpy() * 255).astype("uint8")
        face_np = cv2.resize(face_np, (img_size, img_size))
        faces.append(face_np)

    cap.release()

    # Require at least 80% of target frames to have a usable face detection
    if len(faces) < int(num_frames * 0.8):
        return None

    # Pad by repeating last frame if slightly short, so all sequences are equal length
    while len(faces) < num_frames:
        faces.append(faces[-1])

    return faces[:num_frames]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--celeb_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="frames")
    parser.add_argument("--frames_per_video", type=int, default=16)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--max_videos_per_class", type=int, default=None,
                         help="Optional cap for faster processing if short on time")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Face detector running on: {device}")
    mtcnn = MTCNN(image_size=args.img_size, margin=20, device=device, post_process=True)

    videos = get_video_list(args.celeb_dir)
    print(f"Found {len(videos)} total videos "
          f"({sum(1 for _, l in videos if l == 0)} real, "
          f"{sum(1 for _, l in videos if l == 1)} fake)")

    if len(videos) == 0:
        raise RuntimeError(
            f"No videos found under {args.celeb_dir}. "
            f"Check folder names match Celeb-real/Celeb-synthesis/YouTube-real."
        )

    # Split BY VIDEO (not by frame) to avoid data leakage
    random.seed(42)
    real_videos = [v for v in videos if v[1] == 0]
    fake_videos = [v for v in videos if v[1] == 1]
    random.shuffle(real_videos)
    random.shuffle(fake_videos)

    if args.max_videos_per_class:
        real_videos = real_videos[:args.max_videos_per_class]
        fake_videos = fake_videos[:args.max_videos_per_class]
        print(f"Capped to {len(real_videos)} real + {len(fake_videos)} fake videos")

    def split(vlist):
        n = len(vlist)
        train_end = int(n * 0.7)
        valid_end = int(n * 0.85)
        return vlist[:train_end], vlist[train_end:valid_end], vlist[valid_end:]

    real_train, real_valid, real_test = split(real_videos)
    fake_train, fake_valid, fake_test = split(fake_videos)

    splits = {
        "train": real_train + fake_train,
        "valid": real_valid + fake_valid,
        "test": real_test + fake_test,
    }

    label_names = {0: "real", 1: "fake"}
    total_processed = 0
    total_skipped = 0

    for split_name, split_videos in splits.items():
        print(f"\nProcessing {split_name} split: {len(split_videos)} videos")
        for video_path, label in split_videos:
            video_id = os.path.splitext(os.path.basename(video_path))[0]
            out_dir = os.path.join(args.output_dir, split_name, label_names[label], video_id)

            if os.path.exists(out_dir) and len(os.listdir(out_dir)) == args.frames_per_video:
                continue  # already processed, resume-friendly

            os.makedirs(out_dir, exist_ok=True)

            faces = extract_faces_from_video(
                video_path, mtcnn, num_frames=args.frames_per_video, img_size=args.img_size
            )

            if faces is None:
                total_skipped += 1
                os.rmdir(out_dir) if os.path.isdir(out_dir) and not os.listdir(out_dir) else None
                continue

            for i, face_img in enumerate(faces):
                out_path = os.path.join(out_dir, f"frame_{i:03d}.jpg")
                cv2.imwrite(out_path, cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR))

            total_processed += 1
            if total_processed % 20 == 0:
                print(f"  Processed {total_processed} videos so far "
                      f"({total_skipped} skipped due to detection failure)")

    print(f"\nDone. Total videos processed: {total_processed}, skipped: {total_skipped}")
    print(f"Frame sequences saved under: {args.output_dir}/")


if __name__ == "__main__":
    main()
