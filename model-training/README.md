# Deepfake Detection — Model Training Quick Start
### (Adjusted for GTX 1650 Ti / 4GB VRAM)

You've already completed environment setup and confirmed:
```
True NVIDIA GeForce GTX 1650 Ti
```
So skip straight to the dataset and training steps below.

## 1. Get a dataset — no approval wait

Use the **"140k Real and Fake Faces"** Kaggle dataset — pre-extracted face
images (~4GB), downloads instantly with a Kaggle account. No registration
wait like FaceForensics++/DFDC.

```bash
# Get your Kaggle API key: kaggle.com -> Account -> Create New API Token
# This downloads kaggle.json — place it here:
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Download dataset (run from ~/deepfake_training)
kaggle datasets download -d xhlulu/140k-real-and-fake-faces
unzip 140k-real-and-fake-faces.zip -d dataset/
```

Expected folder structure after unzip:
```
dataset/
  real_vs_fake/
    real-vs-fake/
      train/real/*.jpg   train/fake/*.jpg
      valid/real/*.jpg   valid/fake/*.jpg
      test/real/*.jpg    test/fake/*.jpg
```

If your unzip produces a different structure, run `find dataset -maxdepth 4 -type d`
and paste the output here — I'll adjust the `--data_dir` path accordingly.

## 2. Run training (settings tuned for your 4GB VRAM)

```bash
python3 train.py --data_dir dataset/real_vs_fake/real-vs-fake --epochs 10
```

This uses batch_size=8 with gradient accumulation (effective batch=32) —
tuned specifically to avoid out-of-memory crashes on a 4GB card. Do NOT
increase `--batch_size` without testing first; jumping straight to 32 will
likely crash with `CUDA out of memory`.

**If you still hit `CUDA out of memory`:**
```bash
python3 train.py --data_dir dataset/real_vs_fake/real-vs-fake --epochs 10 --batch_size 4 --accum_steps 8
```
This keeps the same effective batch size (32) with an even smaller physical
batch.

## 3. Expect timing

On a GTX 1650 Ti, expect roughly 15-25 minutes per epoch on the full 140k
dataset (~100k train images). If you're short on time before your defense,
you can:
- Reduce epochs: `--epochs 3` still gives real, honest partial results
- Or subsample the dataset first (see note below) for faster iteration

To subsample for a quicker first run (optional, only if very tight on time):
```bash
# Keep only first 5000 images per class in train/ (adjust path as needed)
cd dataset/real_vs_fake/real-vs-fake/train/real && ls | tail -n +5001 | xargs rm --
cd ../fake && ls | tail -n +5001 | xargs rm --
```
Only do this if you need a fast sanity-check run — for your actual reported
results, prefer the full dataset if time allows.

## 4. What you'll get

- `checkpoints/epoch_N.pt` — a checkpoint after every epoch
- `checkpoints/best_model.pt` — best model by validation accuracy
- `training_curve.png` — real loss/accuracy plot, ready to drop into your
  report or slides
- Terminal output showing real loss/accuracy numbers per epoch — screenshot
  this too, it's good defense evidence

## 5. Honest framing for your defense

If training isn't fully finished: "Training was started on the
140k-real-and-fake-faces dataset. X epochs completed so far, achieving Y%
validation accuracy. Full training on video-based datasets (FaceForensics++/
DFDC) with the temporal LSTM component is planned for Iteration 3." This is
completely normal and defensible at a midterm stage.
