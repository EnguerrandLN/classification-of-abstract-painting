"""
AbstractViz — serveur Flask
  GET  /              → index.html
  GET  /api/data      → JSON de tous les points
  GET  /api/detail/<idx> → voisins + composition cluster
  POST /api/groq      → proxy Groq (vision + fallback texte)
  GET  /assets/...    → images statiques
"""

import os, json, base64, io
import numpy as np
import pandas as pd
import requests
from pathlib import Path
from flask import Flask, jsonify, request, send_from_directory, send_file

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
GROQ_API_KEY  = "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"
ROOT_DIR      = os.path.dirname(os.path.abspath(__file__))
CSV_FILENAME = "resultats_clusters enguerrand pas de titre pour l'insatnt.csv"
DATA_PATH = os.path.join(ROOT_DIR, CSV_FILENAME)

# Auto-détection du sous-dossier dans assets/
_assets_root = os.path.join(ROOT_DIR, "assets")
if os.path.isdir(_assets_root):
    _subs = [d for d in os.listdir(_assets_root) if os.path.isdir(os.path.join(_assets_root, d))]
    ASSETS_SUBFOLDER = _subs[0] if _subs else ""
else:
    ASSETS_SUBFOLDER = "abstrait-v4"
ASSETS_FOLDER = os.path.join(ROOT_DIR, "assets", ASSETS_SUBFOLDER)
print(f"✓ Images : assets/{ASSETS_SUBFOLDER}/ ({len(os.listdir(ASSETS_FOLDER)) if os.path.isdir(ASSETS_FOLDER) else '⚠ INTROUVABLE'} fichiers)")

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

# ─────────────────────────────────────────────
#  DONNÉES
# ─────────────────────────────────────────────
if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
    if "filename" in df.columns and "image_path" not in df.columns:
        df = df.rename(columns={"filename": "image_path"})
    print(f"✓ {len(df)} œuvres chargées depuis '{DATA_PATH}'")
else:
    print(f"⚠ '{DATA_PATH}' introuvable — données simulées.")
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

def cluster_color(cid):
    return CLUSTER_COLORS[int(cid) % len(CLUSTER_COLORS)]

def artist_color(a):
    return ARTIST_PALETTE[hash(a) % len(ARTIST_PALETTE)]

def hex_to_rgb(h):
    h = h.lstrip("#")
    return [int(h[i:i+2], 16) for i in (0, 2, 4)]

# ─────────────────────────────────────────────
#  MINIATURES BASE64 (80px) pour l'API /data
# ─────────────────────────────────────────────
def encode_thumb(rel_path, size=80):
    full = os.path.join(ASSETS_FOLDER, str(rel_path))
    if not os.path.exists(full):
        return None
    try:
        raw = open(full, "rb").read()
        try:
            from PIL import Image as PILImage, ImageDraw
            
            # Convertir en RGBA pour gérer la transparence
            im = PILImage.open(io.BytesIO(raw)).convert("RGBA")
            
            # 1. Recadrer au centre pour avoir un carré parfait
            w, h = im.size
            min_dim = min(w, h)
            left = (w - min_dim) / 2
            top = (h - min_dim) / 2
            im = im.crop((left, top, left + min_dim, top + min_dim))
            
            # 2. Redimensionner
            im.thumbnail((size, size), PILImage.LANCZOS)
            
            # 3. Créer un masque circulaire
            mask = PILImage.new('L', im.size, 0)
            draw = ImageDraw.Draw(mask)
            draw.ellipse((0, 0, im.size[0], im.size[1]), fill=255)
            
            # 4. Appliquer le masque pour rendre les coins transparents
            im.putalpha(mask)
            
            # 5. Sauvegarder obligatoirement en PNG (le JPEG ne supporte pas la transparence)
            buf = io.BytesIO()
            im.save(buf, format="PNG", optimize=True)
            raw = buf.getvalue()
        except Exception:
            pass
        return f"data:image/png;base64,{base64.b64encode(raw).decode()}"
    except Exception:
        return None

print("⏳ Encodage des miniatures…")
thumbs = {}
for _, row in df.iterrows():
    t = encode_thumb(row["image_path"])
    if t:
        thumbs[int(row["index"])] = t
print(f"✓ {len(thumbs)}/{len(df)} miniatures encodées")

# ─────────────────────────────────────────────
#  FLASK APP
# ─────────────────────────────────────────────
app = Flask(__name__, static_folder=None)

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
            "thumb":   thumbs.get(idx),        # base64 miniature ou null
            "cc":      cc,   # cluster color RGB
            "ac":      ac,   # artist color RGB
        })
    meta = {
        "n":             len(df),
        "artists":       ARTISTS,
        "clusters":      CLUSTERS,
        "cluster_colors":{str(c): cluster_color(c) for c in CLUSTERS},
        "artist_colors": {a: artist_color(a) for a in ARTISTS},
        "assets_prefix": f"assets/{ASSETS_SUBFOLDER}",
    }
    return jsonify({"points": points, "meta": meta})

@app.route("/api/detail/<int:idx>")
def api_detail(idx):
    """Voisins proches + composition du cluster."""
    row = df[df["index"] == idx]
    if len(row) == 0:
        return jsonify({"error": "not found"}), 404
    row = row.iloc[0]

    dx   = df["x"] - row["x"]
    dy   = df["y"] - row["y"]
    dist = np.sqrt(dx**2 + dy**2)
    dist = dist.drop(df[df["index"] == idx].index)
    nn   = dist.nsmallest(8).index
    neighbors = []
    for i in nn:
        r = df.loc[i]
        neighbors.append({
            "index":    int(r["index"]),
            "artist":   r["artist"],
            "cluster":  int(r["cluster"]),
            "distance": float(dist[i]),
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

@app.route("/assets/<path:filename>")
def serve_asset(filename):
    return send_from_directory(os.path.join(ROOT_DIR, "assets"), filename)

# ─────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\n{'='*55}")
    print("  AbstractViz — Flask + deck.gl")
    print(f"{'='*55}")
    print(f"  {len(df)} œuvres | {len(ARTISTS)} artistes | {len(CLUSTERS)} classes")
    print(f"  ➜  http://localhost:5000")
    print(f"{'='*55}\n")
    app.run(debug=True, port=5000)
