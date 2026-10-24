"""Phase 4 (map): a static, serverless interactive map -> outputs/map/index.html.

One self-contained HTML file (data inlined, no library, no server: open it from disk) plus
small thumbnails in outputs/map/thumbs/. Full-size images are read from abstrait-v4/ through a
relative path. Run scripts/build_facets.py first. Nothing here is committed: the thumbnails are
derived from copyrighted paintings.
"""

import json
import os

from PIL import Image
from tqdm import tqdm

from abstractart import DATASET_DIR, OUTPUT_DIR

MAP_DIR = OUTPUT_DIR / "map"
THUMB_SIZE = 160


def make_thumbnails(filenames: list[str]) -> None:
    thumbs = MAP_DIR / "thumbs"
    thumbs.mkdir(parents=True, exist_ok=True)
    for f in tqdm(filenames, desc="thumbnails"):
        out = thumbs / f
        if out.exists():
            continue
        img = Image.open(DATASET_DIR / f)
        img.draft("RGB", (2 * THUMB_SIZE, 2 * THUMB_SIZE))
        img = img.convert("RGB")
        img.thumbnail((THUMB_SIZE, THUMB_SIZE), Image.LANCZOS)
        img.save(out, "JPEG", quality=80)


def main():
    data = json.loads((OUTPUT_DIR / "map_data.json").read_text(encoding="utf-8"))
    make_thumbnails([p["f"] for p in data["paintings"]])
    dataset = os.path.relpath(DATASET_DIR, MAP_DIR).replace(os.sep, "/")
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    html = TEMPLATE.replace("__DATA__", payload).replace("__DATASET__", dataset)
    (MAP_DIR / "index.html").write_text(html, encoding="utf-8")
    print(f"{len(data['paintings'])} paintings, views {list(data['views'])} -> {MAP_DIR / 'index.html'}")


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Abstract Art Map</title>
<style>
:root { --bg:#f6f5f2; --panel:#ffffff; --ink:#1d1d1f; --muted:#6b6b70; --line:#e4e2dc; --accent:#3b5bdb; --canvas:#eeede8; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#121214; --panel:#1b1b1f; --ink:#ececf0; --muted:#9a9aa3; --line:#2d2d34; --accent:#8da2fb; --canvas:#161619; }
}
* { box-sizing: border-box; }
html, body { margin: 0; height: 100%; background: var(--bg); color: var(--ink);
  font: 14px/1.45 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }
#app { display: grid; grid-template-columns: 1fr 390px; height: 100vh; }
#stage { position: relative; overflow: hidden; background: var(--canvas); }
canvas { display: block; width: 100%; height: 100%; cursor: grab; touch-action: none; }
canvas.dragging { cursor: grabbing; }
#toolbar { position: absolute; top: 12px; left: 12px; right: 12px; display: flex; gap: 8px; flex-wrap: wrap; pointer-events: none; }
#toolbar > * { pointer-events: auto; }
.tabs { display: flex; background: var(--panel); border: 1px solid var(--line); border-radius: 8px; overflow: hidden; }
.tabs button { border: 0; background: none; color: var(--ink); padding: 7px 13px; font: inherit; cursor: pointer; }
.tabs button.on { background: var(--accent); color: #fff; }
select, input { font: inherit; color: var(--ink); background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 6px 10px; }
#hint { position: absolute; bottom: 10px; left: 12px; font-size: 12px; color: var(--muted); }
#tip { position: absolute; pointer-events: none; background: var(--panel); border: 1px solid var(--line); border-radius: 8px;
  padding: 6px; display: none; width: 190px; box-shadow: 0 6px 20px rgba(0,0,0,.18); }
#tip img { width: 100%; display: block; border-radius: 4px; }
#tip div { margin-top: 4px; font-size: 12px; }
aside { background: var(--panel); border-left: 1px solid var(--line); overflow-y: auto; padding: 16px 18px; }
h1 { font-size: 17px; margin: 0 0 2px; }
h2 { font-size: 13px; text-transform: uppercase; letter-spacing: .04em; color: var(--muted); margin: 18px 0 6px; }
.muted { color: var(--muted); font-size: 12px; }
.legend { list-style: none; padding: 0; margin: 0; }
.legend li { display: flex; gap: 8px; align-items: baseline; padding: 4px 6px; border-radius: 6px; cursor: pointer; }
.legend li:hover, .legend li.on { background: var(--line); }
.sw { width: 10px; height: 10px; border-radius: 50%; flex: none; transform: translateY(1px); }
.legend .n { margin-left: auto; color: var(--muted); font-variant-numeric: tabular-nums; }
#detail img.main { width: 100%; max-height: 340px; object-fit: contain; border-radius: 6px; display: block; background: var(--line); }
.badge { display: inline-block; font-size: 11px; padding: 1px 8px; border-radius: 10px; border: 1px solid var(--line); margin-left: 6px; }
.nbrs { display: grid; grid-template-columns: repeat(4, 1fr); gap: 6px; }
.nbrs img { width: 100%; aspect-ratio: 1; object-fit: cover; border-radius: 4px; cursor: pointer; display: block; }
dl { margin: 6px 0 0; } dt { font-weight: 600; font-size: 12px; margin-top: 8px; } dd { margin: 2px 0 0; font-size: 13px; }
.close { float: right; border: 0; background: none; color: var(--muted); font: inherit; cursor: pointer; }
@media (max-width: 820px) {
  #app { grid-template-columns: 1fr; grid-template-rows: 62vh auto; height: auto; }
  aside { border-left: 0; border-top: 1px solid var(--line); }
}
</style>
</head>
<body>
<div id="app">
  <div id="stage">
    <canvas id="c"></canvas>
    <div id="toolbar">
      <div class="tabs" id="tabs"></div>
      <select id="layer" aria-label="Level of the hierarchy"></select>
      <input id="search" list="artists" placeholder="Find an artist…" aria-label="Find an artist">
      <datalist id="artists"></datalist>
    </div>
    <div id="hint">Scroll to zoom · drag to pan · click a painting · Esc to reset</div>
    <div id="tip"></div>
  </div>
  <aside>
    <h1>Abstract art map</h1>
    <p class="muted" id="about"></p>
    <div id="detail"></div>
    <h2 id="groups-title">Groups</h2>
    <ul class="legend" id="legend"></ul>
    <p class="muted" id="footnote"></p>
  </aside>
</div>
<script id="data" type="application/json">__DATA__</script>
<script>
const DATA = JSON.parse(document.getElementById("data").textContent);
const DATASET = "__DATASET__";
const P = DATA.paintings, N = P.length, VIEWS = DATA.views;
const PALETTE = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f", "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#86bcb6"];
const $ = id => document.getElementById(id);
const canvas = $("c"), ctx = canvas.getContext("2d");
let view = "overall", layerIdx = 0, selected = -1, hover = -1, focusCluster = null, focusArtist = null;
let W = 0, H = 0, DPR = 1, colours = new Map(), order = [];
const cam = { x: 0, y: 0, k: 1 };
const pos = Float32Array.from(VIEWS.overall.xy.flat()), from = new Float32Array(2 * N), to = new Float32Array(2 * N);
let animStart = -1;

const layer = () => VIEWS[view].layers[layerIdx];
const base = () => Math.min(W, H) * 0.46;
const sx = i => W / 2 + (pos[2 * i] - cam.x) * base() * cam.k;
const sy = i => H / 2 + (pos[2 * i + 1] - cam.y) * base() * cam.k;
const esc = s => String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const pretty = s => s.replace(/_/g, " ");

function setupColours() {
  const L = layer(), sizes = {};
  L.labels.forEach(c => sizes[c] = (sizes[c] || 0) + 1);
  order = Object.keys(sizes).map(Number).sort((a, b) => sizes[b] - sizes[a]);
  colours = new Map(order.map((c, r) => [c, order.length <= PALETTE.length ? PALETTE[r]
    : `hsl(${(r * 137.508) % 360} 62% ${r % 2 ? 46 : 58}%)`]));
}

const images = new Map();
function thumb(i) {
  let im = images.get(i);
  if (!im) { im = new Image(); im.onload = requestDraw; im.src = "thumbs/" + encodeURIComponent(P[i].f); images.set(i, im); }
  return im.complete && im.naturalWidth ? im : null;
}

let pending = false;
function requestDraw() { if (!pending) { pending = true; requestAnimationFrame(draw); } }

function draw(now) {
  pending = false;
  if (animStart >= 0) {
    const t = Math.min(1, ((now || performance.now()) - animStart) / 700), e = t < .5 ? 4 * t * t * t : 1 - (-2 * t + 2) ** 3 / 2;
    for (let j = 0; j < 2 * N; j++) pos[j] = from[j] + (to[j] - from[j]) * e;
    if (t < 1) requestDraw(); else animStart = -1;
  }
  ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
  ctx.clearRect(0, 0, W, H);
  const L = layer(), zoom = base() * cam.k, showThumbs = zoom > 1100, ts = Math.max(18, Math.min(72, zoom / 45));
  const ink = getComputedStyle(document.body).getPropertyValue("--ink");
  for (const pass of [0, 1]) for (let i = 0; i < N; i++) {
    const core = L.core[i] === 1;
    if ((pass === 0) === core) continue;  // transitional first, core on top
    const x = sx(i), y = sy(i);
    if (x < -80 || y < -80 || x > W + 80 || y > H + 80) continue;
    const dim = (focusCluster !== null && L.labels[i] !== focusCluster) || (focusArtist && P[i].a !== focusArtist);
    const col = colours.get(L.labels[i]);
    const im = showThumbs && !dim ? thumb(i) : null;
    ctx.globalAlpha = dim ? 0.1 : im ? (core ? 1 : 0.7) : core ? 0.92 : 0.6;
    if (im) {
      const s = ts / Math.max(im.naturalWidth, im.naturalHeight), w = im.naturalWidth * s, h = im.naturalHeight * s;
      ctx.drawImage(im, x - w / 2, y - h / 2, w, h);
      ctx.strokeStyle = col; ctx.lineWidth = core ? 2 : 1; ctx.strokeRect(x - w / 2, y - h / 2, w, h);
    } else {
      ctx.beginPath(); ctx.arc(x, y, core ? 3.4 : 3, 0, 2 * Math.PI);
      if (core) { ctx.fillStyle = col; ctx.fill(); } else { ctx.strokeStyle = col; ctx.lineWidth = 1.4; ctx.stroke(); }
    }
  }
  ctx.globalAlpha = 1;
  if (selected >= 0) {
    ctx.strokeStyle = ink; ctx.lineWidth = 1; ctx.globalAlpha = .45;
    for (const j of VIEWS[view].neighbours[selected]) { ctx.beginPath(); ctx.moveTo(sx(selected), sy(selected)); ctx.lineTo(sx(j), sy(j)); ctx.stroke(); }
    ctx.globalAlpha = 1; ctx.lineWidth = 2.5;
    ctx.beginPath(); ctx.arc(sx(selected), sy(selected), 9, 0, 2 * Math.PI); ctx.stroke();
  }
  if (hover >= 0 && hover !== selected) {
    ctx.strokeStyle = ink; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.arc(sx(hover), sy(hover), 7, 0, 2 * Math.PI); ctx.stroke();
  }
}

function resize() {
  DPR = window.devicePixelRatio || 1; W = canvas.clientWidth; H = canvas.clientHeight;
  canvas.width = Math.round(W * DPR); canvas.height = Math.round(H * DPR); requestDraw();
}

function nearest(mx, my, radius = 9) {
  let best = -1, bd = radius * radius;
  for (let i = 0; i < N; i++) { const dx = sx(i) - mx, dy = sy(i) - my, d = dx * dx + dy * dy; if (d < bd) { bd = d; best = i; } }
  return best;
}

function clusterInfo(v, li, i) {
  const L = VIEWS[v].layers[li];
  return { id: L.labels[i], core: L.core[i] === 1, info: L.clusters[String(L.labels[i])] || {} };
}

function termsText(terms) {
  const list = Array.isArray(terms) ? terms : Object.values(terms || {}).flat();
  return list.sort((a, b) => b[1] - a[1]).slice(0, 4).map(([t, z]) => `${esc(t)} (${z >= 0 ? "+" : ""}${z.toFixed(1)})`).join(", ");
}
const measuredText = m => (m || []).map(([d, z]) => `${esc(pretty(d))} ${z >= 0 ? "+" : ""}${z.toFixed(1)}σ`).join(", ");

function renderDetail() {
  const d = $("detail");
  if (selected < 0) { d.innerHTML = ""; return; }
  const p = P[selected], here = clusterInfo(view, layerIdx, selected);
  const others = Object.keys(VIEWS).filter(v => v !== view).map(v => {
    const c = clusterInfo(v, 0, selected);
    return `<dt>${esc(VIEWS[v].label)} view</dt><dd>${esc(c.info.title || "—")}${c.core ? "" : " <span class='muted'>(transitional)</span>"}</dd>`;
  }).join("");
  d.innerHTML = `
    <h2>Selected painting <button class="close" id="close" aria-label="Close">✕</button></h2>
    <img class="main" src="${DATASET}/${encodeURIComponent(p.f)}" alt="${esc(p.f)}">
    <p><strong>${esc(pretty(p.a))}</strong> <span class="muted">${esc(p.f)}</span>
       <span class="badge">${(layer().core_names || ["transitional", "core"])[here.core ? 1 : 0]}</span></p>
    <dl>
      <dt>${esc(layer().name)}</dt><dd>${esc(here.info.title || "—")}</dd>
      ${layer().sets ? `<dd class="muted">${esc(layer().sets_label)}: ${layer().sets[selected].map(c =>
        esc((layer().clusters[String(c)] || {}).title || "group " + c)).join(" <strong>or</strong> ")}</dd>` : ""}
      ${here.info.terms ? `<dd class="muted">${termsText(here.info.terms)}</dd>` : ""}
      ${here.info.measured && here.info.measured.length ? `<dd class="muted">${measuredText(here.info.measured)}</dd>` : ""}
      ${others}
    </dl>
    <h2>Nearest paintings (${esc(VIEWS[view].label.toLowerCase())} view)</h2>
    <div class="nbrs">${VIEWS[view].neighbours[selected].map(j =>
      `<img src="thumbs/${encodeURIComponent(P[j].f)}" data-i="${j}" title="${esc(pretty(P[j].a))}" alt="${esc(P[j].f)}">`).join("")}</div>`;
  $("close").onclick = () => select(-1);
  d.querySelectorAll(".nbrs img").forEach(im => im.onclick = () => select(+im.dataset.i, true));
}

function renderLegend() {
  const L = layer(), counts = {};
  L.labels.forEach(c => counts[c] = (counts[c] || 0) + 1);
  $("groups-title").textContent = L.name;
  $("footnote").textContent = (L.core_names
    ? "Filled dots: a single plausible family. Rings: several families are plausible at this confidence level."
    : "Filled dots: paintings in a dense region of their group. Rings: transitional paintings, attached to the nearest group.")
    + " Neighbours are computed in the full embedding, not on this 2-D map.";
  $("legend").innerHTML = order.map(c => {
    const info = L.clusters[String(c)] || {};
    return `<li data-c="${c}" class="${focusCluster === c ? "on" : ""}"><span class="sw" style="background:${colours.get(c)}"></span>
      <span>${esc(info.title || "group " + c)}</span><span class="n">${counts[c]}</span></li>`;
  }).join("");
  $("legend").querySelectorAll("li").forEach(li => li.onclick = () => {
    const c = +li.dataset.c; focusCluster = focusCluster === c ? null : c; renderLegend(); requestDraw();
  });
}

function select(i, centre = false) {
  selected = i;
  if (centre && i >= 0) { cam.x = pos[2 * i]; cam.y = pos[2 * i + 1]; }
  renderDetail(); requestDraw();
}

function setView(v) {
  view = v; layerIdx = 0; focusCluster = null;
  from.set(pos); to.set(VIEWS[v].xy.flat()); animStart = performance.now();
  $("tabs").querySelectorAll("button").forEach(b => b.classList.toggle("on", b.dataset.v === v));
  $("layer").innerHTML = VIEWS[v].layers.map((L, k) => `<option value="${k}">${esc(L.name)}</option>`).join("");
  $("layer").style.display = VIEWS[v].layers.length > 1 ? "" : "none";
  $("about").textContent = VIEWS[v].about;
  setupColours(); renderLegend(); renderDetail(); requestDraw();
}

$("tabs").innerHTML = Object.keys(VIEWS).map(v => `<button data-v="${v}">${esc(VIEWS[v].label)}</button>`).join("");
$("tabs").querySelectorAll("button").forEach(b => b.onclick = () => setView(b.dataset.v));
$("layer").onchange = e => { layerIdx = +e.target.value; focusCluster = null; setupColours(); renderLegend(); renderDetail(); requestDraw(); };
$("artists").innerHTML = [...new Set(P.map(p => p.a))].sort().map(a => `<option value="${esc(a)}">`).join("");
$("search").oninput = e => { const a = e.target.value.trim().toLowerCase(); focusArtist = P.some(p => p.a === a) ? a : null; requestDraw(); };

let drag = null;
canvas.addEventListener("pointerdown", e => { drag = { x: e.offsetX, y: e.offsetY, moved: false }; canvas.setPointerCapture(e.pointerId); });
canvas.addEventListener("pointermove", e => {
  if (drag) {
    const dx = e.offsetX - drag.x, dy = e.offsetY - drag.y;
    if (Math.abs(dx) + Math.abs(dy) > 3) drag.moved = true;
    if (drag.moved) { cam.x -= dx / (base() * cam.k); cam.y -= dy / (base() * cam.k); drag.x = e.offsetX; drag.y = e.offsetY;
      canvas.classList.add("dragging"); $("tip").style.display = "none"; requestDraw(); return; }
  }
  const i = nearest(e.offsetX, e.offsetY);
  if (i !== hover) { hover = i; requestDraw(); }
  const tip = $("tip");
  if (i < 0) { tip.style.display = "none"; return; }
  const c = clusterInfo(view, layerIdx, i);
  tip.innerHTML = `<img src="thumbs/${encodeURIComponent(P[i].f)}" alt=""><div><strong>${esc(pretty(P[i].a))}</strong><br>
    <span class="muted">${esc(c.info.title || "")}${c.core ? "" : " · transitional"}</span></div>`;
  tip.style.display = "block";
  tip.style.left = Math.min(e.offsetX + 14, W - 200) + "px"; tip.style.top = Math.min(e.offsetY + 14, H - 240) + "px";
});
canvas.addEventListener("pointerup", e => {
  canvas.classList.remove("dragging");
  if (drag && !drag.moved) select(nearest(e.offsetX, e.offsetY));
  drag = null;
});
canvas.addEventListener("pointerleave", () => { hover = -1; $("tip").style.display = "none"; requestDraw(); });
canvas.addEventListener("wheel", e => {
  e.preventDefault();
  const wx = (e.offsetX - W / 2) / (base() * cam.k) + cam.x, wy = (e.offsetY - H / 2) / (base() * cam.k) + cam.y;
  cam.k = Math.min(60, Math.max(0.6, cam.k * Math.exp(-e.deltaY * 0.0015)));
  cam.x = wx - (e.offsetX - W / 2) / (base() * cam.k); cam.y = wy - (e.offsetY - H / 2) / (base() * cam.k);
  requestDraw();
}, { passive: false });
window.addEventListener("keydown", e => {
  if (e.key === "Escape") { select(-1); focusCluster = null; focusArtist = null; $("search").value = ""; cam.x = cam.y = 0; cam.k = 1; renderLegend(); requestDraw(); }
});
window.addEventListener("resize", resize);
resize(); setView("overall");
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
