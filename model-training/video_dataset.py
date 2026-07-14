"""
Dataset loader for video frame SEQUENCES (for the CNN+LSTM temporal model).
Expects folder structure produced by extract_frames.py:
    frames/
        train/real/<video_id>/frame_000.jpg ... frame_015.jpg
        train/fake/<video_id>/...
        valid/real/... valid/fake/...
        test/real/...  test/fake/...

Each __getitem__ returns a tensor of shape [seq_len, C, H, W] — a full frame
sequence for one video — plus its label.
"""
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class VideoSequenceDataset(Dataset):
    def __init__(self, root_dir, split="train", img_size=224, seq_len=16):
        self.samples = []  # list of (video_folder_path, label)
        self.seq_len = seq_len
        split_dir = os.path.join(root_dir, split)

        for label_name, label_idx in [("real", 0), ("fake", 1)]:
            class_dir = os.path.join(split_dir, label_name)
            if not os.path.isdir(class_dir):
                raise FileNotFoundError(f"Expected folder not found: {class_dir}")

            for video_id in os.listdir(class_dir):
                video_folder = os.path.join(class_dir, video_id)
                if not os.path.isdir(video_folder):
                    continue
                frame_files = sorted(f for f in os.listdir(video_folder) if f.endswith(".jpg"))
                if len(frame_files) >= seq_len:
                    self.samples.append((video_folder, label_idx))

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No video sequences found in {split_dir}. "
                f"Did extract_frames.py finish running for this split?"
            )

        if split == "train":
            self.transform = T.Compose([
                T.Resize((img_size, img_size)),
                T.RandomHorizontalFlip(),
                T.ColorJitter(brightness=0.2, contrast=0.2),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = T.Compose([
                T.Resize((img_size, img_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_folder, label = self.samples[idx]
        frame_files = sorted(f for f in os.listdir(video_folder) if f.endswith(".jpg"))[:self.seq_len]

        frames = []
        for fname in frame_files:
            img = Image.open(os.path.join(video_folder, fname)).convert("RGB")
            img = self.transform(img)
            frames.append(img)

        # Stack into [seq_len, C, H, W]
        sequence = torch.stack(frames, dim=0)
        return sequence, label
