# SortBin

SortBin is a live waste-material classifier that uses a webcam or a DroidCam phone camera and a Keras MobileNetV2 model. It classifies frames as **organic**, **plastic**, **metal**, or **general**.

## Requirements

- Python 3.10–3.12
- A webcam, or a phone running DroidCam

Install the Python dependencies:

```powershell
python -m pip install tensorflow opencv-python numpy
```

## Run

From the repository root, start with the first model and the default camera:

```powershell
python .\sortbin\live_sortbin.py --model .\sortbin\sortbin_mobilenetv2.keras --source 0
```

To use the alternate model, pass `.\sortbin\sortbin_mobilenetv2_v2.keras` to `--model`.

For DroidCam over Wi-Fi, replace the source with the phone's video URL:

```powershell
python .\sortbin\live_sortbin.py --model .\sortbin\sortbin_mobilenetv2.keras --source http://<phone-ip>:4747/video
```

For the DroidCam PC client, use the webcam index shown on your computer (commonly `0`, `1`, or `2`).

## Controls

- `q` quits.
- `o`, `p`, `m`, `g`, and `e` save the current frame as organic, plastic, metal, general, or empty.

Captured frames are saved under `local_data/<class>/` by default. Use `--save-dir` to choose another directory. The prediction confidence threshold and frame smoothing can be adjusted with `--threshold` and `--smooth`.
