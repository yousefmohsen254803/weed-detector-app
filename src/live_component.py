"""Live, on-device detection: the phone's camera feed is processed in the browser.

The YOLO model (exported to ONNX) runs inside the browser with ONNX Runtime Web, so no video
is sent to the server. Works on any https page (Streamlit Cloud) and on localhost.
Pre- and post-processing match Ultralytics: rectangular letterbox (stride 32, grey 114 padding),
confidence threshold, per-class NMS with IoU 0.5.
"""
import json

import streamlit.components.v1 as components

ORT_VERSION = "1.20.1"
MODEL_FILE = "seedling_all.onnx"  # served from src/static/ (server.enableStaticServing = true)

_HTML = r"""
<style>
  body { margin: 0; font-family: "Source Sans Pro", system-ui, sans-serif; color: #31333F; }
  #wrap { position: relative; width: 100%; background: #111; border-radius: 8px; overflow: hidden; }
  #v { width: 100%; display: block; }
  #c { position: absolute; left: 0; top: 0; width: 100%; height: 100%; }
  #banner { padding: 10px 12px; border-radius: 8px; margin: 8px 0; font-weight: 600; background: #eef3fb; }
  #banner.hit { background: #dff3e1; color: #1a6b2a; }
  .row { display: flex; gap: 8px; flex-wrap: wrap; margin: 8px 0; }
  button { font-size: 16px; padding: 10px 16px; border-radius: 8px; border: 1px solid #ccc; background: #fff; }
  button.primary { background: #3cb043; color: #fff; border-color: #3cb043; }
  .stats { display: grid; grid-template-columns: repeat(4, 1fr); gap: 6px; font-size: 13px; }
  .stats div { background: #f6f6f8; border-radius: 6px; padding: 6px; text-align: center; }
  .stats b { display: block; font-size: 20px; }
  #status { font-size: 13px; color: #666; }
</style>
<div class="row">
  <button id="start" class="primary" disabled>Loading model…</button>
  <button id="flip" disabled>Switch camera</button>
</div>
<div id="status">Loading the detector into your browser (≈11 MB, first time only)…</div>
<div id="banner">Camera off</div>
<div id="wrap"><video id="v" playsinline muted></video><canvas id="c"></canvas></div>
<div class="stats">
  <div><b id="s0">0</b>seedlings</div><div><b id="s1">0</b>leaves</div>
  <div><b id="s2">0</b>moss</div><div><b id="fps">–</b>fps</div>
</div>
<script src="https://cdn.jsdelivr.net/npm/onnxruntime-web@__ORT__/dist/ort.min.js"></script>
<script>
const CONF = __CONF__, SHOW = __SHOW__, IOU = 0.5, MAXSIDE = 640;
const NAMES = ["weed_seedling", "fallen_leaf", "moss"];
const COLORS = ["rgb(60,220,60)", "rgb(255,190,0)", "rgb(230,60,230)"];
const $ = id => document.getElementById(id);
const video = $("v"), overlay = $("c"), octx = overlay.getContext("2d");
const work = document.createElement("canvas"), wctx = work.getContext("2d", { willReadFrequently: true });
let session = null, stream = null, facing = "environment", busy = false, running = false, fps = 0, lastT = 0, dets = [];

function modelUrls() {
  const urls = [];
  try { urls.push(new URL("app/static/__MODEL__", document.baseURI).href); } catch (e) {}
  try { urls.push(window.parent.location.origin + "/app/static/__MODEL__"); } catch (e) {}
  urls.push("/app/static/__MODEL__");
  return [...new Set(urls)];
}

async function loadModel() {
  ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@__ORT__/dist/";
  ort.env.wasm.numThreads = 1;
  for (const url of modelUrls()) {
    try {
      const r = await fetch(url);
      if (!r.ok) continue;
      const buf = await r.arrayBuffer();
      session = await ort.InferenceSession.create(buf, { executionProviders: ["wasm"] });
      $("status").textContent = "Detector ready — runs on this device, nothing is uploaded.";
      $("start").disabled = false; $("start").textContent = "📷 Start live scan";
      return;
    } catch (e) { console.warn("model load failed", url, e); }
  }
  $("status").textContent = "Could not load the model file. Check that the app has static serving enabled.";
}

async function startCamera() {
  stopCamera();
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: false, video: { facingMode: { ideal: facing }, width: { ideal: 1280 }, height: { ideal: 720 } } });
  } catch (e) {
    $("status").textContent = "Camera blocked: " + e.message + ". Allow camera access for this site and try again.";
    return;
  }
  video.srcObject = stream; await video.play();
  overlay.width = video.videoWidth; overlay.height = video.videoHeight;
  running = true; $("start").textContent = "⏹ Stop"; $("flip").disabled = false;
  requestAnimationFrame(loop);
}

function stopCamera() {
  running = false;
  if (stream) stream.getTracks().forEach(t => t.stop());
  stream = null; dets = []; octx.clearRect(0, 0, overlay.width, overlay.height);
  $("start").textContent = "📷 Start live scan"; $("banner").className = ""; $("banner").textContent = "Camera off";
}

function letterbox() {
  const w = video.videoWidth, h = video.videoHeight, r = Math.min(MAXSIDE / w, MAXSIDE / h);
  const nw = Math.round(w * r), nh = Math.round(h * r);
  const W = Math.ceil(nw / 32) * 32, H = Math.ceil(nh / 32) * 32;
  const px = Math.floor((W - nw) / 2), py = Math.floor((H - nh) / 2);
  work.width = W; work.height = H;
  wctx.fillStyle = "rgb(114,114,114)"; wctx.fillRect(0, 0, W, H);
  wctx.drawImage(video, px, py, nw, nh);
  const px4 = wctx.getImageData(0, 0, W, H).data, n = W * H, x = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) { x[i] = px4[4 * i] / 255; x[n + i] = px4[4 * i + 1] / 255; x[2 * n + i] = px4[4 * i + 2] / 255; }
  return { tensor: new ort.Tensor("float32", x, [1, 3, H, W]), r, px, py };
}

function iou(a, b) {
  const ix = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0]));
  const iy = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
  const inter = ix * iy, u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
  return u > 0 ? inter / u : 0;
}

function decode(out, r, px, py) {
  const d = out.data, A = out.dims[2], nc = out.dims[1] - 4, cand = [];
  for (let i = 0; i < A; i++) {
    let best = 0, cls = 0;
    for (let c = 0; c < nc; c++) { const s = d[(4 + c) * A + i]; if (s > best) { best = s; cls = c; } }
    if (best < CONF || !SHOW[cls]) continue;
    const cx = d[i], cy = d[A + i], bw = d[2 * A + i], bh = d[3 * A + i];
    cand.push({ cls, conf: best, box: [(cx - bw / 2 - px) / r, (cy - bh / 2 - py) / r, (cx + bw / 2 - px) / r, (cy + bh / 2 - py) / r] });
  }
  cand.sort((a, b) => b.conf - a.conf);
  const keep = [];
  for (const c of cand) {
    if (keep.length >= 300) break;
    if (keep.every(k => k.cls !== c.cls || iou(k.box, c.box) <= IOU)) keep.push(c);
  }
  return keep;
}

function draw() {
  octx.clearRect(0, 0, overlay.width, overlay.height);
  const lw = Math.max(2, overlay.width / 300), fs = Math.max(14, overlay.width / 45);
  octx.font = `bold ${fs}px sans-serif`;
  for (const d of dets) {
    const [x1, y1, x2, y2] = d.box;
    octx.strokeStyle = COLORS[d.cls]; octx.lineWidth = lw; octx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    const label = `${NAMES[d.cls]} ${d.conf.toFixed(2)}`, tw = octx.measureText(label).width;
    const ty = y1 - fs - 4 > 0 ? y1 - fs - 4 : y2;
    octx.fillStyle = COLORS[d.cls]; octx.fillRect(x1, ty, tw + 6, fs + 4);
    octx.fillStyle = "#000"; octx.fillText(label, x1 + 3, ty + fs);
  }
  const n = [0, 1, 2].map(c => dets.filter(d => d.cls === c).length);
  $("s0").textContent = n[0]; $("s1").textContent = n[1]; $("s2").textContent = n[2];
  $("fps").textContent = fps ? fps.toFixed(1) : "–";
  const b = $("banner");
  if (n[0]) { b.className = "hit"; b.textContent = `🌱 ${n[0]} weed seedling(s) in view — robot would remove`; }
  else { b.className = ""; b.textContent = "No weed seedlings in view"; }
}

async function loop() {
  if (!running) return;
  if (!busy && video.readyState >= 2) {
    busy = true;
    try {
      const { tensor, r, px, py } = letterbox();
      const res = await session.run({ [session.inputNames[0]]: tensor });
      dets = decode(res[session.outputNames[0]], r, px, py);
      const now = performance.now(); if (lastT) fps = 0.8 * fps + 0.2 * (1000 / (now - lastT)); lastT = now;
      draw();
    } catch (e) { console.error(e); $("status").textContent = "Detection error: " + e.message; }
    busy = false;
  }
  requestAnimationFrame(loop);
}

$("start").onclick = () => (running ? stopCamera() : startCamera());
$("flip").onclick = () => { facing = facing === "environment" ? "user" : "environment"; if (running) startCamera(); };
loadModel();
</script>
"""


def live_scanner(conf: float, show: list[bool], height: int = 820):
    """Render the on-device live scanner. `show` = [seedling, leaf, moss] visibility."""
    html = (_HTML.replace("__ORT__", ORT_VERSION).replace("__MODEL__", MODEL_FILE)
            .replace("__CONF__", f"{conf:.3f}").replace("__SHOW__", json.dumps([bool(s) for s in show])))
    components.html(html, height=height, scrolling=True)
