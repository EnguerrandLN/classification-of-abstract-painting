import os, json, base64, io
from scipy.spatial import cKDTree
import numpy as np
import pandas as pd
import requests
from pathlib import Path
from flask import Flask, jsonify, request, send_from_directory, send_file
from functools import wraps

# Config
GROQ_API_KEY  = os.environ.get("GROQ_API_KEY", "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")
DATA_PATH     = "resultats_clusters.csv"
ASSETS_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "abstrait-v4")
ROOT_DIR      = os.path.dirname(os.path.abspath(__file__))

CLUSTER_COLORS = [
    "#7F77DD","#1D9E75","#D85A30","#D4537E","#EF9F27",
    "#378ADD","#85B7EB","#5DCAA5","#F0997B","#97C459",
    "#AFA9EC","#BA7517","#A32D2D","#888780","#ED93B1",
]
ARTIST_PALETTE = [
    "#7F77DD","#1D9E75","#D85A30","#D4537E","#EF9F27",
    "#378ADD","#85B7EB","#5DCAA5","#F0997B","#97C459",
    "#AFA9EC","#BA7517","#A32D2D","#ED93B1","#1D6E56",
    "#E85D9A","#2A9F6E","#C0622F","#6A5ACD","#3CB371",
]

# Données
if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
    if "filename" in df.columns and "image_path" not in df.columns:
        df = df.rename(columns={"filename": "image_path"})
    print(f"{len(df)} œuvres chargées depuis '{DATA_PATH}'")
else:
    print(f"'{DATA_PATH}' introuvable — données simulées.")
    np.random.seed(42)
    n = 1700
    centers = [(-20,20),(20,20),(0,-25),(-25,-15),(25,-10),(0,0),
               (-10,10),(10,-10),(30,0),(-30,0),(0,30),(15,25),
               (-15,-25),(25,-25),(-25,25)]
    xs, ys, cl = [], [], []
    for i in range(n):
        c = i % 15
        cx, cy = centers[c]
        xs.append(cx + np.random.randn()*6)
        ys.append(cy + np.random.randn()*6)
        cl.append(c)
    DEMO_ARTISTS = ["bergman","rothko","vasarely","hartung","fontana",
                    "mitchell","zao","soulages","de-stael","kandinsky"]
    df = pd.DataFrame({
        "index": range(n), "x": xs, "y": ys,
        "artist": np.random.choice(DEMO_ARTISTS, n),
        "image_path": [f"work_{i}.jpg" for i in range(n)],
        "cluster": cl,
    })

ARTISTS  = sorted(df["artist"].unique().tolist())
CLUSTERS = sorted(df["cluster"].unique().tolist())

# KD-Tree pour les voisins
_coords   = None
_kdtree   = None

def _build_kdtree():
    global _coords, _kdtree
    _coords = df[["x","y"]].values
    _kdtree = cKDTree(_coords)

def cluster_color(cid):
    return CLUSTER_COLORS[int(cid) % len(CLUSTER_COLORS)]

def artist_color(a):
    return ARTIST_PALETTE[hash(a) % len(ARTIST_PALETTE)]

def hex_to_rgb(h):
    h = h.lstrip("#")
    return [int(h[i:i+2], 16) for i in (0, 2, 4)]

# Miniatures base64 (80px)
def encode_thumb(rel_path, size=80):
    full = os.path.join(ASSETS_FOLDER, str(rel_path))
    if not os.path.exists(full):
        return None
    try:
        raw = open(full, "rb").read()
        try:
            from PIL import Image as PILImage, ImageDraw

            im = PILImage.open(io.BytesIO(raw)).convert("RGBA")

            # crop carré centré
            w, h = im.size
            min_dim = min(w, h)
            left = (w - min_dim) / 2
            top = (h - min_dim) / 2
            im = im.crop((left, top, left + min_dim, top + min_dim))

            im.thumbnail((size, size), PILImage.LANCZOS)

            # masque circulaire
            mask = PILImage.new('L', im.size, 0)
            draw = ImageDraw.Draw(mask)
            draw.ellipse((0, 0, im.size[0], im.size[1]), fill=255)
            im.putalpha(mask)

            buf = io.BytesIO()
            im.save(buf, format="PNG", optimize=True)
            raw = buf.getvalue()
        except Exception:
            pass
        return f"data:image/png;base64,{base64.b64encode(raw).decode()}"
    except Exception:
        return None

_build_kdtree()
print(f"KD-Tree construit ({len(df)} nœuds)")

print("Encodage des miniatures…")
thumbs = {}
for _, row in df.iterrows():
    t = encode_thumb(row["image_path"])
    if t:
        thumbs[int(row["index"])] = t
print(f"{len(thumbs)}/{len(df)} miniatures encodées")

# Flask app
app = Flask(__name__, static_folder=None)

@app.after_request
def add_cors(r):
    r.headers["Access-Control-Allow-Origin"]  = "*"
    r.headers["Access-Control-Allow-Headers"] = "Content-Type"
    r.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return r

@app.route("/api/data",   methods=["OPTIONS"])
@app.route("/api/groq",   methods=["OPTIONS"])
@app.route("/api/detail", methods=["OPTIONS"])
def _options(): return "", 204

@app.route("/")
def index():
    return send_file(os.path.join(ROOT_DIR, "index.html"))

@app.route("/api/data")
def api_data():
    """Retourne tous les points avec couleurs RGB précalculées."""
    points = []
    for _, row in df.iterrows():
        idx = int(row["index"])
        cid = int(row["cluster"])
        art = row["artist"]
        cc  = hex_to_rgb(cluster_color(cid))
        ac  = hex_to_rgb(artist_color(art))
        points.append({
            "i":       idx,
            "x":       float(row["x"]),
            "y":       float(row["y"]),
            "artist":  art,
            "cluster": cid,
            "path":    str(row["image_path"]),
            "thumb":   thumbs.get(idx),
            "cc":      cc,
            "ac":      ac,
        })
    meta = {
        "n":        len(df),
        "artists":  ARTISTS,
        "clusters": CLUSTERS,
        "cluster_colors": {str(c): cluster_color(c) for c in CLUSTERS},
        "artist_colors":  {a: artist_color(a) for a in ARTISTS},
    }
    return jsonify({"points": points, "meta": meta})

@app.route("/api/detail/<int:idx>")
def api_detail(idx):
    """Voisins proches + composition du cluster."""
    row = df[df["index"] == idx]
    if len(row) == 0:
        return jsonify({"error": "not found"}), 404
    row = row.iloc[0]

    dists, pos = _kdtree.query([row["x"], row["y"]], k=9)
    neighbors = []
    for d_val, p in zip(dists[1:], pos[1:]):   # skip self
        r = df.iloc[p]
        neighbors.append({
            "index":    int(r["index"]),
            "artist":   r["artist"],
            "cluster":  int(r["cluster"]),
            "distance": float(d_val),
            "x":        float(r["x"]),
            "y":        float(r["y"]),
        })

    cid        = int(row["cluster"])
    cluster_df = df[df["cluster"] == cid]
    counts     = cluster_df["artist"].value_counts()
    total      = len(cluster_df)
    composition = [
        {"artist": a, "count": int(c), "pct": round(c/total*100, 1)}
        for a, c in counts.items()
    ]

    same_ac    = df[(df["artist"]==row["artist"]) & (df["cluster"]==cid)]
    same_total = df[df["artist"]==row["artist"]]
    pct_ac     = round(len(same_ac)/max(len(same_total),1)*100, 1)

    return jsonify({
        "index":        idx,
        "artist":       row["artist"],
        "cluster":      cid,
        "path":         str(row["image_path"]),
        "same_ac":      len(same_ac),
        "same_total":   len(same_total),
        "pct_ac":       pct_ac,
        "neighbors":    neighbors,
        "composition":  composition,
        "cluster_color": cluster_color(cid),
        "artist_color":  artist_color(row["artist"]),
    })

@app.route("/api/groq", methods=["POST"])
def api_groq():
    """Proxy Groq : vision llama-4-scout avec fallback texte."""
    body      = request.get_json()
    artist    = body.get("artist", "")
    img_path  = body.get("image_path", "")
    artist_cl = artist.replace("-"," ").replace("_"," ").title()

    full_path = os.path.join(ASSETS_FOLDER, img_path)
    if not os.path.exists(full_path):
        return jsonify({"error": f"Image introuvable : {img_path}"}), 404

    raw = open(full_path, "rb").read()
    try:
        from PIL import Image
        img_obj = Image.open(io.BytesIO(raw))
        img_obj.thumbnail((800, 800), Image.LANCZOS)
        buf = io.BytesIO()
        img_obj.save(buf, format="JPEG", quality=82)
        raw = buf.getvalue()
    except Exception:
        pass

    img_b64 = base64.standard_b64encode(raw).decode()
    prompt  = (
        f"Tu es un expert en histoire de l'art. Voici un tableau abstrait attribué à {artist_cl}.\n\n"
        "Réponds en français avec exactement ce format (4 lignes) :\n\n"
        "**Artiste :** [Prénom Nom, nationalité, années de vie]\n"
        "**Titre probable :** [titre le plus vraisemblable ou \"Sans titre\"]\n"
        "**Date approximative :** [période estimée, ex: \"vers 1955\"]\n"
        "**Analyse :** [2-3 phrases : couleurs, formes, composition, énergie]"
    )
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}

    try:
        r = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, timeout=30, json={
            "model": "meta-llama/llama-4-scout-17b-16e-instruct",
            "max_tokens": 400,
            "messages": [{"role":"user","content":[
                {"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{img_b64}"}},
                {"type":"text","text":prompt}
            ]}]
        })
        r.raise_for_status()
        return jsonify({"result": r.json()["choices"][0]["message"]["content"]})
    except Exception:
        pass

    # fallback texte
    try:
        r2 = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, timeout=20, json={
            "model": "llama-3.3-70b-versatile",
            "max_tokens": 400,
            "messages": [{"role":"user","content":(
                f"Tu es un expert en histoire de l'art. Pour une œuvre abstraite de {artist_cl}, "
                "réponds en français :\n\n"
                "**Artiste :** ...\n**Titre probable :** ...\n**Date approximative :** ...\n**Analyse :** ..."
            )}]
        })
        r2.raise_for_status()
        txt = r2.json()["choices"][0]["message"]["content"]
        return jsonify({"result": txt + "\n\n_⚠ Analyse visuelle indisponible._"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/thumbs", methods=["POST"])
def api_thumbs_batch():
    """Batch lazy: reçoit indices, retourne miniatures base64 (cache serveur)."""
    indices = (request.get_json() or {}).get("indices", [])
    out = {}
    for idx in indices[:50]:
        idx = int(idx)
        if idx not in thumbs:
            row = df[df["index"] == idx]
            if len(row):
                t = encode_thumb(row.iloc[0]["image_path"])
                if t: thumbs[idx] = t
        out[idx] = thumbs.get(idx)
    return jsonify(out)

@app.route("/api/lasso", methods=["POST"])
def api_lasso():
    """Points dans un rectangle data."""
    b = request.get_json() or {}
    xmin,ymin,xmax,ymax = b.get("xmin",0),b.get("ymin",0),b.get("xmax",0),b.get("ymax",0)
    mask = (df["x"]>=xmin)&(df["x"]<=xmax)&(df["y"]>=ymin)&(df["y"]<=ymax)
    sub  = df[mask]
    return jsonify({
        "n":       int(len(sub)),
        "indices": sub["index"].tolist(),
        "artists": sub["artist"].value_counts().to_dict(),
        "clusters":{int(k):int(v) for k,v in sub["cluster"].value_counts().items()},
    })

@app.route("/assets/<path:filename>")
def serve_asset(filename):
    return send_from_directory(os.path.join(ROOT_DIR, "assets"), filename)

if __name__ == "__main__":
    print(f"{len(df)} œuvres | {len(ARTISTS)} artistes | {len(CLUSTERS)} classes")
    print("http://localhost:5000")
    app.run(debug=True, port=5000)