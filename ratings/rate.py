"""Rate a batch of voices made by `cvsynth sample`. See GENOME_VOICE_DESIGN.md §1.

    python ratings/rate.py ratings/batches/r001 [--port 8765] [--rater NAME]

Opens a local page that plays each sample and records a score from -1 to 1 for how much you like it
as the voice of the creature shown, plus an optional wrong-size tick:
- "Would suit a bigger animal" / "Would suit a smaller animal": tick one only when you'd probably
  like the sound on a creature of another size. The score is still for this creature, so a ticked
  sample usually scores low; the tick says why.

Ratings are appended to ratings/ratings.jsonl (next to this script), one line per rating, never
rewritten. A changed rating is a new line; the newest line for an id wins. The page resumes at the
first unrated sample. Only the standard library is used.
"""
import argparse
import datetime
import json
import math
import os
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlparse

HERE = os.path.dirname(os.path.abspath(__file__))
RATINGS = os.path.join(HERE, "ratings.jsonl")
_write_lock = threading.Lock()

# About-this-size comparisons, by body shape: (mass in kg, "a ...").
SIZES = {
    "quadruped": [(0.02, "a mouse"), (0.3, "a rat"), (1.5, "a rabbit"), (4, "a cat"), (8, "a fox"), (25, "a dog"),
                  (70, "a person"), (150, "a deer"), (350, "a bear"), (700, "a horse")],
    "bird": [(0.02, "a sparrow"), (0.08, "a thrush"), (0.4, "a crow"), (1.2, "a duck"), (4, "a goose"),
             (12, "a swan")],
    "bug": [(0.001, "a cricket"), (0.005, "a big beetle"), (0.05, "a mouse"), (0.5, "a rat"), (3, "a cat")],
    "serpent": [(0.3, "a small snake"), (2, "a grass snake"), (10, "a python"), (60, "a large python")],
}


def load_manifest(folder):
    path = os.path.join(folder, "manifest.jsonl")
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def latest_ratings():
    out = {}
    if os.path.exists(RATINGS):
        with open(RATINGS, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    out[r["id"]] = r
    return out


def describe(sample):
    """What the page shows about a sample: the creature, not its settings."""
    b = sample["body"]
    shape = b.get("shape", "quadruped")
    mass = b["mass"]
    like = min(SIZES.get(shape, SIZES["quadruped"]), key=lambda s: abs(math.log(mass / s[0])))[1]
    return {
        "id": sample["id"],
        "shape": shape,
        "mass": mass,
        "like": like,
        "intent": sample["intent"],
        "takes": sample["takes"],
    }


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Voice Ratings</title>
<style>
:root { --bg:#f6f5f2; --panel:#fff; --ink:#1d1d1b; --muted:#6b6a66; --line:#dddbd5; --accent:#2f6fdb; --good:#2e8b57; --bad:#c0392b; --tick:#b8860b; }
@media (prefers-color-scheme: dark) { :root { --bg:#161615; --panel:#1f1f1d; --ink:#ecebe7; --muted:#9a9993; --line:#33332f; --accent:#6d9cf0; --good:#5cbf86; --bad:#e0685a; --tick:#e0b04a; } }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--ink); font:16px/1.45 system-ui, -apple-system, Segoe UI, sans-serif; }
main { max-width:640px; margin:0 auto; padding:24px 16px 48px; }
header { display:flex; justify-content:space-between; align-items:baseline; margin-bottom:16px; }
h1 { font-size:18px; margin:0; }
.progress { color:var(--muted); font-variant-numeric: tabular-nums; }
.card { background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:20px; }
.creature { display:flex; gap:20px; align-items:flex-end; }
svg.scene { width:220px; height:150px; flex:none; }
svg.scene .human { fill:none; stroke:var(--muted); stroke-width:2; }
svg.scene .animal { fill:var(--ink); }
svg.scene .ground { stroke:var(--line); stroke-width:1; }
.facts { min-width:0; }
.facts .big { font-size:22px; font-weight:600; }
.facts .muted { color:var(--muted); }
.row { display:flex; gap:10px; align-items:center; flex-wrap:wrap; margin-top:18px; }
button { font:inherit; border:1px solid var(--line); background:var(--panel); color:var(--ink); border-radius:8px; padding:8px 12px; cursor:pointer; }
button:hover { border-color:var(--muted); }
button.primary { background:var(--accent); border-color:var(--accent); color:#fff; }
button.on { background:var(--tick); border-color:var(--tick); color:#000; }
kbd { font:12px ui-monospace, monospace; border:1px solid var(--line); border-radius:4px; padding:0 4px; color:var(--muted); }
.slider { width:100%; margin-top:6px; }
input[type=range] { width:100%; accent-color:var(--accent); }
.marks { display:flex; justify-content:space-between; color:var(--muted); font-size:13px; font-variant-numeric: tabular-nums; }
.score { font-size:28px; font-weight:700; font-variant-numeric: tabular-nums; min-width:70px; }
input[type=text] { font:inherit; width:100%; padding:8px 10px; border:1px solid var(--line); border-radius:8px; background:var(--bg); color:var(--ink); }
.hint { color:var(--muted); font-size:13px; margin-top:14px; }
.done { text-align:center; padding:40px 0; }
.status { color:var(--muted); font-size:13px; min-height:1.4em; margin-top:8px; }
</style></head>
<body><main>
<header><h1>Voice ratings</h1><span class="progress" id="progress"></span></header>
<div id="start" class="card done"><p>Each voice plays when it opens. Rate how much you like it <b>as the voice of the creature shown</b>.</p>
<button class="primary" id="go">Start</button></div>
<div id="card" class="card" hidden>
  <div class="creature">
    <svg class="scene" id="scene" viewBox="0 0 220 150" aria-label="creature next to a person for scale"></svg>
    <div class="facts">
      <div class="big" id="like"></div>
      <div class="muted" id="mass"></div>
      <div class="muted" id="intent"></div>
    </div>
  </div>
  <div class="row"><button id="play">Play <kbd>Space</kbd></button><span class="muted" id="takes"></span></div>
  <div class="row" style="align-items:baseline"><span class="score" id="scoreText">0.0</span><span class="muted">how much you like it on this creature</span></div>
  <div class="slider"><input type="range" id="score" min="-1" max="1" step="0.1" value="0">
    <div class="marks"><span>−1 <kbd>1</kbd></span><span>−0.5 <kbd>2</kbd></span><span>0 <kbd>3</kbd></span><span>0.5 <kbd>4</kbd></span><span>1 <kbd>5</kbd></span></div></div>
  <div class="row">
    <button id="bigger">Would suit a bigger animal <kbd>B</kbd></button>
    <button id="smaller">Would suit a smaller animal <kbd>S</kbd></button>
  </div>
  <div class="row"><input type="text" id="note" placeholder="Note (optional)"></div>
  <div class="row">
    <button id="back">Back <kbd>⌫</kbd></button>
    <button id="skip">Skip <kbd>X</kbd></button>
    <span style="flex:1"></span>
    <button class="primary" id="next">Save and next <kbd>Enter</kbd></button>
  </div>
  <div class="status" id="status"></div>
  <div class="hint">Tick a size button only when you'd probably like this sound on a creature of another size. Arrow keys nudge the score by 0.1.</div>
</div>
<div id="finished" class="card done" hidden><p><b>All rated.</b> Thank you.</p><p class="muted" id="summary"></p><button id="review">Go back to the last one</button></div>
</main>
<script>
const $ = id => document.getElementById(id);
let samples = [], rated = {}, i = 0, size = null, listens = 0, audio = null, playing = false, busy = false;

// Rough body length in metres from mass, by shape, and a human of 1.75 m for scale.
function lengthOf(shape, mass) {
  const c = Math.cbrt(mass);
  return {quadruped: 0.3 * c, bird: 0.32 * c, bug: 0.35 * c, serpent: 1.1 * c}[shape] || 0.3 * c;
}
function silhouette(shape) {
  // Unit-length outlines (x 0..1, y up from 0), drawn scaled.
  switch (shape) {
    case "bird": return "M0.05,0.55 C0.1,0.8 0.45,0.85 0.6,0.65 L0.72,0.78 L0.95,0.72 L0.72,0.66 L0.7,0.5 C0.6,0.3 0.2,0.25 0.05,0.55 Z M0.35,0.32 L0.33,0 L0.36,0 L0.4,0.3 Z M0.48,0.3 L0.47,0 L0.5,0 L0.53,0.3 Z";
    case "bug": return "M0.15,0.25 C0.15,0.55 0.75,0.6 0.85,0.35 C0.95,0.15 0.25,0.0 0.15,0.25 Z M0.3,0.2 L0.18,0 M0.5,0.18 L0.5,0 M0.7,0.2 L0.82,0 M0.3,0.4 L0.12,0.55 M0.85,0.35 L1.0,0.5";
    case "serpent": return "M0,0.05 C0.15,0.25 0.25,0.25 0.35,0.08 C0.45,-0.05 0.55,-0.05 0.65,0.1 C0.75,0.25 0.85,0.25 0.95,0.12 L1.0,0.14 L0.98,0.2 C0.85,0.33 0.72,0.3 0.62,0.15 C0.53,0.04 0.47,0.04 0.39,0.15 C0.27,0.33 0.12,0.32 0,0.12 Z";
    default: return "M0.1,0.45 C0.12,0.62 0.6,0.65 0.75,0.55 L0.88,0.68 L1.0,0.62 L0.95,0.5 L0.8,0.45 C0.78,0.4 0.75,0.35 0.75,0.3 L0.73,0 L0.68,0 L0.65,0.3 L0.3,0.3 L0.25,0 L0.2,0 L0.18,0.32 C0.12,0.35 0.1,0.4 0.1,0.45 Z M0.1,0.45 L0.0,0.35";
  }
}
function drawScene(shape, mass) {
  const L = lengthOf(shape, mass), H = 1.75;
  const W = 220, Hpx = 150, ground = 140, pad = 10;
  // Fit both the person (0.5 m wide) and the creature side by side.
  const scale = Math.min((W - 3 * pad) / (0.5 + L), (ground - pad) / Math.max(H, L * 0.7));
  const s = Math.max(scale, 1e-6);
  const hx = pad, hw = 0.5 * s, hh = H * s;
  const human = `<g class="human"><circle cx="${hx + hw / 2}" cy="${ground - hh + hh * 0.07}" r="${hh * 0.07}"/>` +
    `<path d="M${hx + hw / 2},${ground - hh * 0.86} L${hx + hw / 2},${ground - hh * 0.45} M${hx + hw / 2},${ground - hh * 0.45} L${hx + hw * 0.15},${ground} M${hx + hw / 2},${ground - hh * 0.45} L${hx + hw * 0.85},${ground} M${hx},${ground - hh * 0.7} L${hx + hw},${ground - hh * 0.7}"/></g>`;
  const cx = hx + hw + pad, cl = Math.max(L * s, 2);
  const path = silhouette(shape).replace(/(-?[\d.]+),(-?[\d.]+)/g, (m, x, y) => `${(cx + x * cl).toFixed(2)},${(ground - y * cl).toFixed(2)}`);
  const stroke = shape === "bug" ? ` stroke="currentColor" stroke-width="${Math.max(cl * 0.03, 0.6)}"` : "";
  const dot = cl < 6 ? `<circle cx="${cx + cl / 2}" cy="${ground - 3}" r="3" class="animal"/>` : "";
  $("scene").innerHTML = `<line class="ground" x1="0" x2="${W}" y1="${ground}" y2="${ground}"/>${human}<path class="animal"${stroke} d="${path}"/>${dot}`;
}
function fmtMass(m) { return m < 1 ? `${Math.round(m * 1000)} g` : m < 10 ? `${m.toFixed(1)} kg` : `${Math.round(m)} kg`; }
const intentWords = {idle: "idle call", mating: "mating call", warn: "warning growl", alarm: "alarm call", flee: "fleeing cry", pain: "pain cry", beg: "begging call", territorial: "territorial call", speak: "speech"};

function setScore(v) { v = Math.max(-1, Math.min(1, Math.round(v * 10) / 10)); $("score").value = v; $("scoreText").textContent = (v > 0 ? "+" : "") + v.toFixed(1); $("scoreText").style.color = v > 0.05 ? "var(--good)" : v < -0.05 ? "var(--bad)" : "var(--ink)"; }
function setSize(v) { size = v; $("bigger").classList.toggle("on", v === "bigger"); $("smaller").classList.toggle("on", v === "smaller"); }
function status(t) { $("status").textContent = t; }

function play() {
  const s = samples[i]; if (!s) return;
  if (audio) { audio.pause(); }
  listens++;
  let k = 0;
  const next = () => {
    if (k >= s.takes.length) { playing = false; return; }
    audio = new Audio("audio/" + encodeURIComponent(s.takes[k++]));
    audio.onended = () => setTimeout(next, 350);
    audio.play().catch(e => { if (e.name !== "AbortError") status("Couldn't play: " + e.message); });
  };
  playing = true; next();
}

function show() {
  const done = Object.keys(rated).length;
  $("progress").textContent = `${Math.min(i + 1, samples.length)} / ${samples.length}`;
  if (i >= samples.length) { $("card").hidden = true; $("finished").hidden = false; const skips = Object.values(rated).filter(r => r.skip).length; $("summary").textContent = `${done} rated, ${skips} skipped. Ratings are in ratings/ratings.jsonl.`; return; }
  $("finished").hidden = true; $("card").hidden = false;
  const s = samples[i], r = rated[s.id];
  drawScene(s.shape, s.mass);
  $("like").textContent = `About the size of ${s.like}`;
  $("mass").textContent = `${fmtMass(s.mass)}, ${s.shape}`;
  $("intent").textContent = intentWords[s.intent] || s.intent;
  $("takes").textContent = s.takes.length > 1 ? "two takes, played one after the other" : "";
  setScore(r && !r.skip ? r.score : 0); setSize(r ? r.size : null); $("note").value = r ? (r.note || "") : "";
  status(r ? (r.skip ? "Skipped before." : "Rated before; saving again replaces it.") : "");
  listens = 0; play();
}

async function save(skip) {
  const s = samples[i];
  const body = {id: s.id, score: skip ? null : parseFloat($("score").value), size: skip ? null : size, skip: !!skip, note: $("note").value.trim(), listens};
  const res = await fetch("rate", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)});
  if (!res.ok) { status("Couldn't save: " + await res.text()); return false; }
  rated[s.id] = body; return true;
}
// One save at a time: a key pressed while a save is in flight would otherwise land on the wrong sample.
async function next(skip) {
  if (busy || i >= samples.length) return;
  busy = true;
  try { if (await save(skip)) { i++; show(); } } finally { busy = false; }
}
function back() { if (!busy && i > 0) { i--; show(); } }

$("go").onclick = () => { $("start").hidden = true; show(); };
$("play").onclick = play;
$("score").oninput = e => setScore(parseFloat(e.target.value));
$("bigger").onclick = () => setSize(size === "bigger" ? null : "bigger");
$("smaller").onclick = () => setSize(size === "smaller" ? null : "smaller");
$("next").onclick = () => next(false);
$("skip").onclick = () => next(true);
$("back").onclick = back;
$("review").onclick = () => { i = samples.length - 1; show(); };
document.addEventListener("keydown", e => {
  if ($("card").hidden) { if (e.key === "Enter" && !$("start").hidden) $("go").click(); return; }
  if (document.activeElement === $("note")) { if (e.key === "Enter") { e.preventDefault(); next(false); } return; }
  const k = e.key.toLowerCase();
  if (k === " ") { e.preventDefault(); play(); }
  else if ("12345".includes(k) && k.length === 1) setScore([-1, -0.5, 0, 0.5, 1]["12345".indexOf(k)]);
  else if (k === "arrowleft" || k === "arrowdown") { e.preventDefault(); setScore(parseFloat($("score").value) - 0.1); }
  else if (k === "arrowright" || k === "arrowup") { e.preventDefault(); setScore(parseFloat($("score").value) + 0.1); }
  else if (k === "b") setSize(size === "bigger" ? null : "bigger");
  else if (k === "s") setSize(size === "smaller" ? null : "smaller");
  else if (k === "x") next(true);
  else if (k === "enter") { e.preventDefault(); next(false); }
  else if (k === "backspace") { e.preventDefault(); back(); }
});
fetch("samples").then(r => r.json()).then(d => {
  samples = d.samples; rated = d.rated;
  i = samples.findIndex(s => !(s.id in rated)); if (i < 0) i = samples.length;
  $("progress").textContent = `${Object.keys(rated).length} of ${samples.length} rated`;
});
</script></body></html>
"""


def make_handler(folder, samples, rater):
    allowed = {t for s in samples for t in s["takes"]}
    shown = [describe(s) for s in samples]
    ids = {s["id"] for s in samples}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, code, body, ctype):
            data = body.encode("utf-8") if isinstance(body, str) else body
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            path = unquote(urlparse(self.path).path)
            if path == "/":
                return self.send(200, PAGE, "text/html; charset=utf-8")
            if path == "/samples":
                rated = {k: v for k, v in latest_ratings().items() if k in ids}
                return self.send(200, json.dumps({"samples": shown, "rated": rated}), "application/json")
            if path.startswith("/audio/"):
                name = path[len("/audio/"):]
                if name not in allowed:
                    return self.send(404, "not found", "text/plain")
                with open(os.path.join(folder, name), "rb") as f:
                    return self.send(200, f.read(), "audio/wav")
            self.send(404, "not found", "text/plain")

        def do_POST(self):
            if urlparse(self.path).path != "/rate":
                return self.send(404, "not found", "text/plain")
            try:
                r = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                if r.get("id") not in ids:
                    raise ValueError("unknown sample id")
                skip = bool(r.get("skip"))
                score = None if skip else max(-1.0, min(1.0, float(r["score"])))
                size = r.get("size") if r.get("size") in ("bigger", "smaller") else None
                line = {
                    "id": r["id"], "score": score, "size": None if skip else size, "skip": skip,
                    "note": str(r.get("note", ""))[:500], "listens": int(r.get("listens", 0)),
                    "time": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "rater": rater,
                }
            except (ValueError, KeyError, TypeError) as e:
                return self.send(400, str(e), "text/plain")
            with _write_lock, open(RATINGS, "a", encoding="utf-8") as f:
                f.write(json.dumps(line) + "\n")
            self.send(200, "ok", "text/plain")

    return Handler


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("batch", help="a batch folder made by cvsynth sample (holds manifest.jsonl)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--rater", default="me", help="stored with each rating")
    ap.add_argument("--no-browser", action="store_true", help="don't open the page")
    ap.add_argument("--ratings", help="ratings file (default: ratings.jsonl next to this script)")
    a = ap.parse_args()
    if a.ratings:
        global RATINGS
        RATINGS = os.path.abspath(a.ratings)
    folder = os.path.abspath(a.batch)
    samples = load_manifest(folder)
    server = ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(folder, samples, a.rater))
    url = f"http://127.0.0.1:{a.port}/"
    rated = sum(1 for k in latest_ratings() if k in {s["id"] for s in samples})
    print(f"{len(samples)} samples, {rated} already rated. Rating page: {url}  (Ctrl+C to stop)")
    if not a.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    sys.exit(main())
