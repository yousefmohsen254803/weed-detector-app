# Weed seedling detector — demo app

Streamlit demo for the BUas × Uptime Robotics Weed Robot challenge (Block A).
A YOLO11n model detects **weed seedlings** (robot removes), **fallen leaves** and **moss** (robot ignores)
in top-down photos of pavement.

> Trained on synthetic images. On real phone photos it finds about 1 in 4 seedlings (recall 0.25),
> so expect misses on a real street.

## Use it
- **Detect** tab: upload a photo, take one with your phone (Browse files → Take photo), or pick an example.
- **Robot scan** tab: **Live scan** runs the model on the live camera feed *inside the browser*
  (ONNX Runtime Web, nothing is uploaded), **Snapshot** takes single photos, **Replay** plays the examples.

Hold the camera straight down, about 10–15 cm above the pavement, like the robot's camera.

## Run locally
```
pip install -r requirements.txt
streamlit run src/demo_app.py
```

## Deploy on Streamlit Community Cloud
1. Push this folder to a GitHub repository.
2. Go to https://share.streamlit.io, sign in with GitHub, click **Create app**.
3. Pick the repository, branch `main`, main file path **`src/demo_app.py`**, then **Deploy**.

## Contents
| Path | What |
|---|---|
| `src/demo_app.py` | the Streamlit app |
| `src/common.py` | class names, colours, box drawing |
| `src/live_component.py` | live in-browser scanner (camera + ONNX Runtime Web) |
| `src/static/seedling_all.onnx` | the same model exported to ONNX for the browser |
| `weights/seedling_all.pt` | trained YOLO11n weights (5.5 MB) |
| `samples/` | 10 synthetic test images + 4 real phone photos (`real_…`) |
| `requirements.txt`, `packages.txt` | Python and system packages for the cloud |
