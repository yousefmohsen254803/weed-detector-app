"""Upload-and-detect demo for the weed-seedling detector.

    streamlit run src/demo_app.py

Tab 1 "Detect": upload a photo (or pick a test image) -> boxes, classes, confidences,
and where the robot would act.
Tab 2 "Robot scan": live camera scan (webcam or phone-as-webcam), browser snapshots,
or a replay of the held-out test frames.
"""
import os, sys, time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CLASSES, ROOT, draw_boxes

# Visual assumption from the generation prompts: ~14 cm of ground across a 1254 px frame.
GROUND_WIDTH_MM = 140

st.set_page_config(page_title="Weed seedling detector", page_icon="🌱", layout="wide")


def running_locally():
    """True when the page is opened on the computer that runs the app (localhost)."""
    if os.environ.get("CODER"):
        return False
    try:
        host = st.context.headers.get("Host", "")
    except Exception:
        return True
    return host.split(":")[0] in {"localhost", "127.0.0.1", "::1", ""}


ON_SERVER = not running_locally()  # cloud, uni server, or a phone on the network
st.title("Weed seedling detector — demo")
st.warning("Trained on **synthetic** images. On real phone photos it finds about 1 in 4 seedlings "
           "(recall 0.25), so expect misses on a real street.")

with st.sidebar:
    _cands = [ROOT / "outputs" / "weights" / "best.pt", ROOT / "weights" / "seedling_all.pt"]
    weights = st.text_input("Model weights", str(next((p for p in _cands if p.exists()), _cands[0])))
    conf = st.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)
    show = st.multiselect("Show classes", CLASSES, default=CLASSES)
    st.caption("Green = weed seedling (robot acts) · orange = fallen leaf · magenta = moss (robot ignores)")


@st.cache_resource
def load(path):
    from ultralytics import YOLO
    return YOLO(path)


def detect(model, img_bgr):
    r = model.predict(img_bgr, conf=conf, imgsz=640, verbose=False)[0]
    out = []
    for b, c, s in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy()):
        if CLASSES[int(c)] in show:
            out.append({"class_id": int(c), "xyxy": b.tolist(), "conf": float(s)})
    return out


def table(dets, w):
    mm = GROUND_WIDTH_MM / w
    rows = []
    for d in sorted(dets, key=lambda d: -d["conf"]):
        x1, y1, x2, y2 = d["xyxy"]
        rows.append({"class": CLASSES[d["class_id"]], "conf": round(d["conf"], 2),
                     "x mm*": round((x1 + x2) / 2 * mm), "y mm*": round((y1 + y2) / 2 * mm),
                     "action": "remove" if d["class_id"] == 0 else "ignore"})
    return pd.DataFrame(rows)


try:
    model = load(weights)
except Exception as e:
    st.error(f"Could not load weights: {e}")
    st.stop()

tab1, tab2 = st.tabs(["Detect", "Robot scan"])
test_dir = ROOT / "samples" if (ROOT / "samples").exists() else ROOT / "dataset" / "images" / "test"
samples = sorted(p for p in test_dir.glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"}) if test_dir.exists() else []

with tab1:
    up = st.file_uploader("Upload a top-down paving photo", type=["png", "jpg", "jpeg"])
    pick = st.selectbox("…or pick an example image (real_… = real phone photo)", ["(none)"] + [p.name for p in samples])
    img = None
    if up is not None:
        img = cv2.imdecode(np.frombuffer(up.read(), np.uint8), cv2.IMREAD_COLOR)
    elif pick != "(none)":
        img = cv2.imread(str(test_dir / pick))
    if img is not None:
        t0 = time.time()
        dets = detect(model, img)
        ms = (time.time() - t0) * 1000
        c1, c2 = st.columns([3, 2])
        with c1:
            st.image(cv2.cvtColor(draw_boxes(img.copy(), dets), cv2.COLOR_BGR2RGB), use_container_width=True)
        with c2:
            n_s = sum(d["class_id"] == 0 for d in dets)
            st.metric("Weed seedlings found", n_s)
            st.caption(f"{len(dets)} objects in total · inference {ms:.0f} ms")
            if dets:
                st.dataframe(table(dets, img.shape[1]), hide_index=True, use_container_width=True)
                st.caption(f"*Position assumes the frame covers ~{GROUND_WIDTH_MM} mm of ground (uncalibrated).")
            else:
                st.info("Nothing detected above the threshold.")

with tab2:
    modes = ["Snapshot (browser camera)", "Replay test images"]
    if not ON_SERVER:  # a live webcam only works on the computer running the app
        modes.insert(0, "Live camera")
    mode = st.radio("Source", modes, horizontal=True)

    # ---------------------------------------------------------------- live camera
    if mode == "Live camera":
        st.write("Hold the camera **straight down** at the pavement, about 10–15 cm above it (like the robot), "
                 "and walk slowly along the joints. To use your phone, run it as a webcam (e.g. Camo) "
                 "and pick its camera number.")
        cc1, cc2, cc3 = st.columns(3)
        cam_idx = cc1.number_input("Camera number", 0, 5, 0, 1,
                                   help="0 = built-in webcam. Try 1 or 2 for a phone (Camo) or USB camera.")
        every = cc2.slider("Detect every n-th frame", 1, 10, 2,
                           help="Higher = smoother video on a slow laptop, but slower to react.")
        running = cc3.toggle("📷 Start scan", key="cam_on")

        alert_box = st.empty()
        frame_box = st.empty()
        ph = [c.empty() for c in st.columns(4)]

        if not running:
            frame_box.info("Switch on **📷 Start scan** to open the camera. Switch it off to stop.")
        else:
            if sys.platform == "win32":
                cap = cv2.VideoCapture(int(cam_idx), cv2.CAP_DSHOW)
            else:
                cap = cv2.VideoCapture(int(cam_idx))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            if not cap.isOpened():
                st.error(f"Could not open camera {cam_idx}. Close other apps that use the camera "
                         "(Teams, Zoom, Camera app) or try another camera number.")
            else:
                n = checked = flagged = 0
                d = []
                last_t, fps = time.time(), 0.0
                try:
                    while True:
                        ok, frame = cap.read()
                        if not ok:
                            st.warning("The camera stopped sending frames.")
                            break
                        n += 1
                        if n == 1 or n % every == 0:
                            d = detect(model, frame)
                            checked += 1
                            if any(x["class_id"] == 0 for x in d):
                                flagged += 1
                        now = time.time()
                        fps = 0.9 * fps + 0.1 / max(now - last_t, 1e-3)
                        last_t = now

                        n_seed = sum(x["class_id"] == 0 for x in d)
                        n_leaf = sum(x["class_id"] == 1 for x in d)
                        n_moss = sum(x["class_id"] == 2 for x in d)
                        if n_seed:
                            alert_box.success(f"🌱 **{n_seed} weed seedling(s) in view — robot would remove**")
                        else:
                            alert_box.info("No weed seedlings in view")

                        shown = draw_boxes(frame.copy(), d)
                        if shown.shape[1] > 960:  # smaller image = smoother stream
                            s = 960 / shown.shape[1]
                            shown = cv2.resize(shown, None, fx=s, fy=s)
                        frame_box.image(cv2.cvtColor(shown, cv2.COLOR_BGR2RGB), use_container_width=True)

                        ph[0].metric("Seedlings in view", n_seed)
                        ph[1].metric("Leaves / moss in view (ignored)", f"{n_leaf} / {n_moss}")
                        ph[2].metric("Frames with seedlings", f"{flagged} / {checked}")
                        ph[3].metric("Speed", f"{fps:.1f} fps")
                finally:
                    cap.release()

    # ---------------------------------------------------------------- browser snapshot
    elif mode == "Snapshot (browser camera)":
        st.write("Point the camera **straight down** at the pavement (about 10–15 cm above it) and take a photo. "
                 "On a phone, switch to the back camera if the selfie camera opens — or use **Detect → Browse files → "
                 "Take photo**, which always uses the back camera.")
        if "shots" not in st.session_state:
            st.session_state.shots = {"photos": 0, "with_seedlings": 0, "seedlings": 0}
        shot = st.camera_input("Take a photo of the pavement")
        if shot is not None:
            img = cv2.imdecode(np.frombuffer(shot.getvalue(), np.uint8), cv2.IMREAD_COLOR)
            dets = detect(model, img)
            n_s = sum(d["class_id"] == 0 for d in dets)
            key = hash(shot.getvalue())
            if st.session_state.get("last_shot") != key:  # count each photo once
                st.session_state.last_shot = key
                st.session_state.shots["photos"] += 1
                st.session_state.shots["seedlings"] += n_s
                st.session_state.shots["with_seedlings"] += int(n_s > 0)
            c1, c2 = st.columns([3, 2])
            with c1:
                st.image(cv2.cvtColor(draw_boxes(img.copy(), dets), cv2.COLOR_BGR2RGB), use_container_width=True)
            with c2:
                st.metric("Weed seedlings in this photo", n_s)
                t = st.session_state.shots
                st.write(f"**Photos taken:** {t['photos']} · **with seedlings:** {t['with_seedlings']} · "
                         f"**seedlings flagged:** {t['seedlings']}")
                if dets:
                    st.dataframe(table(dets, img.shape[1]), hide_index=True, use_container_width=True)
        if st.button("Reset tally"):
            st.session_state.shots = {"photos": 0, "with_seedlings": 0, "seedlings": 0}
            st.session_state.pop("last_shot", None)

    # ---------------------------------------------------------------- replay (old demo)
    else:
        st.write("Plays the example frames in sequence, as if the robot drives over them.")
        speed = st.slider("Seconds per frame", 0.2, 3.0, 0.8, 0.1)
        if st.button("Start replay", disabled=not samples):
            frame_box, stats_box = st.empty(), st.empty()
            tally = {c: 0 for c in CLASSES}
            for i, p in enumerate(samples, 1):
                im = cv2.imread(str(p))
                d = detect(model, im)
                for x in d:
                    tally[CLASSES[x["class_id"]]] += 1
                frame_box.image(cv2.cvtColor(draw_boxes(im, d), cv2.COLOR_BGR2RGB), caption=f"frame {i}/{len(samples)}", width=640)
                stats_box.write(f"**Frames scanned:** {i} · **seedlings flagged:** {tally['weed_seedling']} · "
                                f"leaves ignored: {tally['fallen_leaf']} · moss ignored: {tally['moss']}")
                time.sleep(speed)
