"""Live SortBin classifier using a DroidCam phone camera (runs on your PC, not Colab).

Setup (Python 3.10 to 3.12):
    pip install tensorflow opencv-python numpy

Run (Wi-Fi mode, phone and PC on the same network):
    python live_sortbin.py --model sortbin_mobilenetv2.keras --source http://192.168.1.50:4747/video

Run (DroidCam PC client installed, appears as a webcam; try 0, 1 or 2):
    python live_sortbin.py --model sortbin_mobilenetv2.keras --source 1

Keys while running:
    q = quit
    o / p / m / g / e = save the current frame as organic / plastic / metal / general / empty
                        into local_data/<class>/ (for your local dataset)
"""
import argparse
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

CLASSES = ['organic', 'plastic', 'metal', 'general']   # same order as training
IMG = (224, 224)
SAVE_KEYS = {ord('o'): 'organic', ord('p'): 'plastic', ord('m'): 'metal',
             ord('g'): 'general', ord('e'): 'empty'}


def open_source(src):
    return cv2.VideoCapture(int(src) if src.isdigit() else src)


def preprocess(frame_bgr):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, IMG, interpolation=cv2.INTER_LINEAR)
    return np.expand_dims(rgb.astype(np.float32), 0)  # raw 0-255, the model rescales internally


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--source', default='0', help='camera index or DroidCam URL')
    ap.add_argument('--smooth', type=int, default=5, help='number of frames to average')
    ap.add_argument('--threshold', type=float, default=0.6, help='below this the label shows as unsure')
    ap.add_argument('--save-dir', default='local_data')
    args = ap.parse_args()

    model = tf.keras.models.load_model(args.model)
    cap = open_source(args.source)
    if not cap.isOpened():
        raise SystemExit('Could not open the camera source')

    history = deque(maxlen=args.smooth)
    last, fps, saved_msg, saved_until = time.time(), 0.0, '', 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        probs = model(preprocess(frame), training=False).numpy()[0]
        history.append(probs)
        avg = np.mean(history, axis=0)
        k = int(avg.argmax())
        label = CLASSES[k] if avg[k] >= args.threshold else 'unsure'

        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - last, 1e-6))
        last = now

        view = frame.copy()
        cv2.putText(view, f'{label} ({avg[k]:.0%})', (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 3)
        for i, (c, p) in enumerate(zip(CLASSES, avg)):
            y = 70 + i * 28
            cv2.rectangle(view, (10, y), (10 + int(p * 200), y + 18), (255, 200, 0), -1)
            cv2.putText(view, f'{c} {p:.0%}', (220, y + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(view, f'{fps:.0f} FPS', (10, view.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if now < saved_until:
            cv2.putText(view, saved_msg, (10, view.shape[0] - 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        cv2.imshow('SortBin live (q to quit)', view)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key in SAVE_KEYS:
            cls = SAVE_KEYS[key]
            out = Path(args.save_dir) / cls
            out.mkdir(parents=True, exist_ok=True)
            path = out / f'{int(now * 1000)}.jpg'
            cv2.imwrite(str(path), frame)  # saves the raw full-resolution frame
            saved_msg, saved_until = f'saved as {cls}', now + 1.5

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
