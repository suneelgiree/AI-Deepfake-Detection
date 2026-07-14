"""
Dataset loader for face-image deepfake classification.
Expects folder structure:
    data_dir/
        train/real/*.jpg   train/fake/*.jpg
        valid/real/*.jpg   valid/fake/*.jpg
        test/real/*.jpg    test/fake/*.jpg
"""
import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


class DeepfakeFaceDataset(Dataset):
    def __init__(self, root_dir, split="train", img_size=224):
        self.samples = []  # list of (path, label) — label 0=real, 1=fake
        split_dir = os.path.join(root_dir, split)

        for label_name, label_idx in [("real", 0), ("fake", 1)]:
            class_dir = os.path.join(split_dir, label_name)
            if not os.path.isdir(class_dir):
                raise FileNotFoundError(
                    f"Expected folder not found: {class_dir}\n"
                    f"Check your dataset unzip structure matches README.md"
                )
            for fname in os.listdir(class_dir):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append((os.path.join(class_dir, fname), label_idx))

        if len(self.samples) == 0:
            raise RuntimeError(f"No images found in {split_dir}. Check dataset path.")

        # Standard normalization matches ImageNet-pretrained EfficientNet expectations
        if split == "train":
            self.transform = T.Compose([
                T.Resize((img_size, img_size)),
                T.RandomHorizontalFlip(),
                T.RandomRotation(10),
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
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        image = self.transform(image)
        return image, label
