# Lip Reading: Japanese Mouth Shape Recognition

Recognises Japanese mouth shapes (**a, i, u, e, o, N**) from face images. The pipeline is MediaPipe Face Mesh landmarks, then face alignment, then a lip ROI crop, then a PyTorch CNN classifier.

## How it works

1. **Landmark detection:** MediaPipe Face Mesh finds 468 facial landmarks.
2. **Head pose:** `solvePnP` on key landmarks estimates pitch, yaw and roll.
3. **Normalisation:** the face is rotated to level the eyes and scaled to a fixed eye distance (160 px), centred on the nose.
4. **Lip ROI:** a 160×160 crop is taken around the mouth corners.
5. **Classification:** a trained PyTorch model predicts the mouth shape. The script draws the ROI and a probability bar chart onto the output frame.

## Files

| File | Purpose |
|---|---|
| `faceestimetion_kuchiyomi.py` | Full pipeline: alignment, ROI extraction, prediction, annotated output images |
| `test.py` | Landmark-detection check over a folder of frames |

## Run

    pip install opencv-python mediapipe torch torchvision pillow numpy
    python faceestimetion_kuchiyomi.py

- Put input frames (`.png`/`.jpg`) in `word_frame/`. Annotated results are written to `processed_images_word_1/`.
- A trained model (`.pth`) is required. Set `model_path` in the script to point to it.

## Stack

Python · PyTorch · OpenCV · MediaPipe
