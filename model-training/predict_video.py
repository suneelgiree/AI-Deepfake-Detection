"""
Run the trained CNN+LSTM deepfake detector on a single video file.

Usage:
    python3 predict_video.py --video path/to/video.mp4 --checkpoint checkpoints_video/best_model.pt
"""
import argparse
import torch
import torch.nn.functional as F
import cv2
from facenet_pytorch import MTCNN
import torchvision.transforms as T

from model_video import CNNLSTMClassifier


def extract_faces_for_inference(video_path, mtcnn, num_frames=8, img_size=224):
    """Same sampling/face-crop logic as extract_frames.py, but returns
    ready-to-use face crops for a single video (no saving to disk)."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        cap.release()
        raise RuntimeError(f"Could not read video (0 frames found): {video_path}")

    if total_frames < num_frames:
        print(f"WARNING: video only has {total_frames} frames, fewer than "
              f"the {num_frames} the model was trained on. Results may be less reliable.")
        frame_indices = list(range(total_frames))
    else:
        frame_indices = [int(i * total_frames / num_frames) for i in range(num_frames)]

    faces = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            face_crop = mtcnn(frame_rgb)
        except Exception:
            face_crop = None

        if face_crop is None:
            continue

        face_np = (face_crop.permute(1, 2, 0).numpy() * 255).astype("uint8")
        face_np = cv2.resize(face_np, (img_size, img_size))
        faces.append(face_np)

    cap.release()

    if len(faces) == 0:
        raise RuntimeError(
            f"No faces detected in {video_path}. Try a different video, "
            f"or check the video actually contains a visible face."
        )

    # Pad by repeating last frame if we got fewer faces than requested
    while len(faces) < num_frames:
        faces.append(faces[-1])

    return faces[:num_frames]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True, help="Path to video file")
    parser.add_argument("--checkpoint", type=str, default="checkpoints_video/best_model.pt")
    parser.add_argument("--seq_len", type=int, default=8,
                         help="Must match seq_len used during training")
    parser.add_argument("--img_size", type=int, default=224)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading face detector...")
    mtcnn = MTCNN(image_size=args.img_size, margin=20, device=device, post_process=True)

    print(f"Extracting {args.seq_len} face frames from: {args.video}")
    faces = extract_faces_for_inference(args.video, mtcnn, num_frames=args.seq_len,
                                          img_size=args.img_size)
    print(f"Successfully extracted {len(faces)} face frames")

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    frame_tensors = [transform(face) for face in faces]
    sequence = torch.stack(frame_tensors, dim=0).unsqueeze(0)  # [1, seq_len, C, H, W]
    sequence = sequence.to(device)

    print("Loading trained model...")
    model = CNNLSTMClassifier(num_classes=2, pretrained=False).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    with torch.no_grad():
        outputs = model(sequence)
        probs = F.softmax(outputs, dim=1)[0]
        pred_class = torch.argmax(probs).item()

    real_prob = probs[0].item()
    fake_prob = probs[1].item()
    verdict = "FAKE" if pred_class == 1 else "REAL"
    confidence = max(real_prob, fake_prob) * 100

    print("\n" + "=" * 50)
    print(f"VIDEO: {args.video}")
    print(f"VERDICT: {verdict}")
    print(f"Confidence: {confidence:.2f}%")
    print(f"  (Real probability: {real_prob*100:.2f}% | Fake probability: {fake_prob*100:.2f}%)")
    print("=" * 50)


if __name__ == "__main__":
    main()
