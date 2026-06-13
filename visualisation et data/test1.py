import dash
from dash import dcc, html, Input, Output, State, callback_context
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import json
import os
import base64
import requests
from pathlib import Path

# clé Groq (https://console.groq.com/keys)
GROQ_API_KEY = "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"

# chargement des données
DATA_PATH = "embeddings.csv"

if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
    if 'filename' in df.columns and 'image_path' not in df.columns:
        df = df.rename(columns={'filename': 'image_path'})
    print(f"Données chargées : {len(df)} œuvres depuis '{DATA_PATH}'")
else:
    print(f"'{DATA_PATH}' introuvable — données simulées.")
    np.random.seed(42)
    n_items = 1700
    centers = [(-20,20),(20,20),(0,-25),(-25,-15),(25,-10),(0,0),
               (-10,10),(10,-10),(30,0),(-30,0),(0,30),(15,25),
               (-15,-25),(25,-25),(-25,25)]
    xs, ys, cl = [], [], []
    for i in range(n_items):
        c = i % 15
        cx, cy = centers[c]
        xs.append(cx + np.random.randn()*6)
        ys.append(cy + np.random.randn()*6)
        cl.append(c)
    ARTISTS_DEMO = ['bergman','rothko','vasarely','hartung','fontana',
                    'mitchell','zao','soulages','de-stael','kandinsky']
    df = pd.DataFrame({
        'index': range(n_items),
        'x': xs, 'y': ys,
        'artist': np.random.choice(ARTISTS_DEMO, n_items),
        'image_path': [f"work_{i}.jpg" for i in range(n_items)],
        'cluster': cl
    })

ARTISTS = sorted(df['artist'].unique().tolist())
CLUSTERS = sorted(df['cluster'].unique().tolist())
n_items = len(df)

CLUSTER_COLORS = [
    '#7F77DD','#1D9E75','#D85A30','#D4537E','#EF9F27',
    '#378ADD','#85B7EB','#5DCAA5','#F0997B','#97C459',
    '#AFA9EC','#BA7517','#A32D2D','#888780','#ED93B1',
]

# cache base64 des images, encodées au démarrage pour le canvas overlay
IMAGES_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets', 'abstrait-v4')

def _encode_img(rel_path):
    full = os.path.join(IMAGES_FOLDER, rel_path)
    if not os.path.exists(full):
        return None
    try:
        with open(full, 'rb') as f:
            raw = f.read()
        # Réduire en thumbnail 80px (rapide, léger)
        try:
            from PIL import Image as PILImage
            import io as _io
            im = PILImage.open(_io.BytesIO(raw)).convert('RGB')
            im.thumbnail((80, 80), PILImage.LANCZOS)
            buf = _io.BytesIO()
            im.save(buf, format='JPEG', quality=75)
            raw = buf.getvalue()
        except Exception:
            pass
        ext = Path(full).suffix.lower().lstrip('.')
        mime = {'jpg':'jpeg','jpeg':'jpeg','png':'png','webp':'webp'}.get(ext,'jpeg')
        return f"data:image/{mime};base64,{base64.b64encode(raw).decode()}"
    except Exception:
        return None

print("Encodage des images pour le canvas overlay…")
IMAGE_B64_MAP = {}  # {str(index): "data:image/jpeg;base64,..."}
for _, row in df.iterrows():
    enc = _encode_img(str(row['image_path']))
    if enc:
        IMAGE_B64_MAP[str(int(row['index']))] = enc
print(f"{len(IMAGE_B64_MAP)}/{n_items} images encodées")

# Sérialisation JSON pour injection dans le HTML (variable JS globale)
IMAGE_B64_JSON = json.dumps(IMAGE_B64_MAP)

# fonctions utilitaires
def get_cluster_color(cluster_id):
    return CLUSTER_COLORS[int(cluster_id) % len(CLUSTER_COLORS)]

def get_artist_color(artist):
    _PALETTE = [
        '#7F77DD','#1D9E75','#D85A30','#D4537E','#EF9F27',
        '#378ADD','#85B7EB','#5DCAA5','#F0997B','#97C459',
        '#AFA9EC','#BA7517','#A32D2D','#ED93B1','#1D6E56',
        '#E85D9A','#2A9F6E','#C0622F','#6A5ACD','#3CB371',
    ]
    return _PALETTE[hash(artist) % len(_PALETTE)]

def compute_nearest_neighbors(idx, n=8):
    row = df[df['index'] == idx].iloc[0]
    dx = df['x'] - row['x']
    dy = df['y'] - row['y']
    distances = np.sqrt(dx**2 + dy**2)
    distances = distances.drop(df[df['index'] == idx].index)
    nearest_idx = distances.nsmallest(n).index
    result = []
    for i in nearest_idx:
        r = df.loc[i]
        result.append({
            'index': int(r['index']),
            'artist': r['artist'],
            'cluster': int(r['cluster']),
            'distance': float(distances[i]),
            'x': float(r['x']),
            'y': float(r['y'])
        })
    return result

def compute_artist_in_cluster(cluster_id):
    cluster_df = df[df['cluster'] == cluster_id]
    counts = cluster_df['artist'].value_counts()
    total = len(cluster_df)
    return [(artist, int(count), round(count / total * 100, 1)) for artist, count in counts.items()]

def call_groq(artist, image_path):
    """Envoie l'image en base64 à llama-4-scout (vision Groq) et retourne titre, date, artiste, analyse."""
    if GROQ_API_KEY == "VOTRE_CLE_GROQ_ICI":
        return "⚠ Clé Groq non configurée. Ajoutez votre clé dans GROQ_API_KEY en haut du fichier."

    artist_clean = artist.replace('-', ' ').replace('_', ' ').title()

    import base64
    assets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets', 'abstrait-v4')
    img_full_path = os.path.join(assets_dir, image_path)

    if not os.path.exists(img_full_path):
        return f"⚠ Image introuvable : {img_full_path}"

    with open(img_full_path, 'rb') as f:
        img_bytes = f.read()

    # Redimensionner si Pillow est disponible (évite les timeouts)
    try:
        from PIL import Image
        import io
        img_obj = Image.open(io.BytesIO(img_bytes))
        img_obj.thumbnail((800, 800), Image.LANCZOS)
        buf = io.BytesIO()
        img_obj.save(buf, format='JPEG', quality=82)
        img_bytes = buf.getvalue()
    except ImportError:
        pass

    img_b64 = base64.standard_b64encode(img_bytes).decode('utf-8')

    # prompt: titre, date, artiste, analyse
    prompt = f"""Tu es un expert en histoire de l'art. Voici un tableau abstrait attribué à {artist_clean}.

Réponds en français avec exactement ce format (4 lignes, pas de liste à puces) :

**Artiste :** [Prénom Nom, nationalité, années de vie]
**Titre probable :** [titre le plus vraisemblable de cette œuvre, ou "Sans titre" si inconnu, basé sur ce que tu vois et tes connaissances de cet artiste]
**Date approximative :** [période ou année estimée, ex: "vers 1955" ou "années 1960"]
**Analyse :** [2-3 phrases sur ce que tu vois visuellement : couleurs dominantes, formes, composition, énergie, ce que l'œuvre exprime ou évoque]"""

    # appel API Groq vision
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "meta-llama/llama-4-scout-17b-16e-instruct",
        "max_tokens": 400,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                {"type": "text", "text": prompt}
            ]
        }]
    }

    try:
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers, json=payload, timeout=30
        )
        resp.raise_for_status()
        return resp.json()['choices'][0]['message']['content']

    except requests.exceptions.Timeout:
        return "⏱ Délai dépassé — réessayez."

    except requests.exceptions.HTTPError:
        # Fallback texte si vision non disponible
        try:
            prompt_text = (
                f"Tu es un expert en histoire de l'art. Pour une œuvre abstraite de {artist_clean}, "
                f"réponds en français avec ce format exact :\n\n"
                f"**Artiste :** [Prénom Nom, nationalité, années de vie]\n"
                f"**Titre probable :** [un titre typique de cet artiste]\n"
                f"**Date approximative :** [période caractéristique de son œuvre]\n"
                f"**Analyse :** [2-3 phrases sur son style visuel : couleurs, formes, composition, énergie]"
            )
            payload_fallback = {
                "model": "llama-3.3-70b-versatile",
                "max_tokens": 400,
                "messages": [{"role": "user", "content": prompt_text}]
            }
            resp2 = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers, json=payload_fallback, timeout=20
            )
            resp2.raise_for_status()
            return resp2.json()['choices'][0]['message']['content'] + "\n\n_⚠ Analyse visuelle indisponible — modèle vision non accessible sur ce compte._"
        except Exception as e2:
            return f"Erreur API (fallback) : {str(e2)[:120]}"

    except Exception as e:
        return f"Erreur : {str(e)[:120]}"


# app dash
app = dash.Dash(
    __name__,
    title="AbstractViz — Analyse Non Supervisée",
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    assets_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets'),
    suppress_callback_exceptions=True  # nécessaire pour les composants créés dynamiquement (ai-btn, ai-content)
)

# options du dropdown de recherche
dropdown_options = [
    {'label': f"#{row['index']} — {row['artist']} ({row['image_path']})", 'value': row['index']}
    for _, row in df.sort_values('index').iterrows()
]

# css
app.index_string = '''
<!DOCTYPE html>
<html>
<head>
    {%metas%}
    <title>{%title%}</title>
    {%favicon%}
    {%css%}
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap" rel="stylesheet">
    <style>
        * { box-sizing: border-box; }
        body { margin:0; background:#0f0f13; color:#e8e6f0; font-family:'Inter',sans-serif; }
        ::-webkit-scrollbar { width:4px; }
        ::-webkit-scrollbar-track { background:#1a1a24; }
        ::-webkit-scrollbar-thumb { background:#3a3a50; border-radius:2px; }

        /* ── Header ── */
        #header { display:flex; align-items:center; justify-content:space-between; padding:14px 28px; border-bottom:.5px solid #2a2a38; }
        #header h1 { font-size:15px; font-weight:500; letter-spacing:.05em; color:#c8c4e8; margin:0; }
        #header span { font-size:12px; color:#5a5878; font-weight:300; }

        /* ── Search bar ── */
        #search-wrapper { padding:10px 28px; border-bottom:.5px solid #1e1e2c; display:flex; align-items:center; gap:12px; }
        #search-wrapper label { font-size:12px; color:#6a6888; white-space:nowrap; }

        /* ── Filter bar ── */
        #filter-wrapper { padding:8px 28px; border-bottom:.5px solid #1e1e2c; display:flex; align-items:center; gap:8px; flex-wrap:wrap; }
        .artist-btn { font-size:11px; padding:4px 10px; border-radius:20px; border:.5px solid #3a3a50; background:transparent; color:#8886a8; cursor:pointer; transition:all .15s; font-family:'Inter',sans-serif; }
        .artist-btn:hover { border-color:#7F77DD; color:#c8c4e8; }

        /* ── Mode info ── */
        .mode-info { padding:8px 18px; background:#1a1a30; border-bottom:.5px solid #2a2a48; font-size:11px; color:#7F77DD; display:none; }
        .mode-info.visible { display:flex; align-items:center; gap:6px; }

        /* ── Main layout ── */
        #main-layout { display:flex; height:calc(100vh - 128px); }

        /* ── Graph column ── */
        #graph-col { flex:1; min-width:0; position:relative; }

        /* ── Canvas overlay (images sur les points) ── */
        #img-canvas {
            position:absolute;
            top:0; left:0;
            pointer-events:none;
            z-index:10;
        }

        /* ── Detail panel ── */
        #detail-panel { width:320px; min-width:320px; background:#12121a; border-left:.5px solid #2a2a38; display:flex; flex-direction:column; overflow-y:auto; }
        #detail-panel .panel-placeholder { flex:1; display:flex; flex-direction:column; align-items:center; justify-content:center; padding:40px 20px; text-align:center; }
        #detail-panel .panel-placeholder p { font-size:13px; color:#4a4868; line-height:1.6; font-weight:300; }
        #detail-panel .panel-placeholder .icon { font-size:36px; margin-bottom:16px; opacity:.3; }

        .detail-img-wrapper { width:100%; background:#0a0a10; display:flex; align-items:center; justify-content:center; padding:20px; border-bottom:.5px solid #2a2a38; }
        .detail-img-wrapper img { max-width:100%; max-height:240px; object-fit:contain; border-radius:4px; }

        .detail-meta { padding:16px 18px; border-bottom:.5px solid #1e1e2c; }
        .detail-meta h3 { font-size:14px; font-weight:500; color:#c8c4e8; margin:0 0 10px; }
        .meta-row { display:flex; justify-content:space-between; font-size:12px; padding:4px 0; border-bottom:.5px solid #1e1e2c; }
        .meta-row:last-child { border-bottom:none; }
        .meta-label { color:#5a5878; }
        .meta-value { color:#a8a6c8; font-weight:500; }
        .cluster-dot { width:6px; height:6px; border-radius:50%; }

        /* ── AI section ── */
        .ai-section { padding:14px 18px; border-bottom:.5px solid #1e1e2c; }
        .ai-section h4 { font-size:12px; font-weight:500; color:#6a6888; letter-spacing:.08em; text-transform:uppercase; margin:0 0 10px; display:flex; align-items:center; gap:6px; }
        .ai-section h4 span.ai-badge { background:#2a1f4a; color:#9f8fe8; font-size:10px; padding:2px 7px; border-radius:10px; letter-spacing:.05em; font-weight:500; text-transform:none; }
        .ai-text { font-size:12px; color:#a8a4c4; line-height:1.7; font-weight:300; }
        .ai-loading { font-size:12px; color:#5a5878; font-style:italic; display:flex; align-items:center; gap:8px; }
        .ai-loading::before { content:''; display:inline-block; width:10px; height:10px; border:1.5px solid #5a5878; border-top-color:#9f8fe8; border-radius:50%; animation:spin .8s linear infinite; }
        @keyframes spin { to { transform:rotate(360deg); } }
        .ai-btn { margin-top:8px; padding:6px 14px; background:#1e1640; border:.5px solid #4a3a80; color:#9f8fe8; border-radius:6px; font-size:11px; cursor:pointer; font-family:'Inter',sans-serif; transition:all .15s; }
        .ai-btn:hover { background:#2a1f5a; border-color:#7F77DD; }

        /* ── Neighbors ── */
        .neighbors-section { padding:14px 18px; }
        .neighbors-section h4 { font-size:12px; font-weight:500; color:#6a6888; letter-spacing:.08em; text-transform:uppercase; margin:0 0 10px; }
        .neighbor-item { display:flex; align-items:center; gap:10px; padding:6px 4px; border-bottom:.5px solid #1a1a24; cursor:pointer; border-radius:4px; transition:background .1s; }
        .neighbor-item:last-child { border-bottom:none; }
        .neighbor-item:hover { background:#1a1a24; }
        .neighbor-rank { font-size:10px; color:#3a3858; width:16px; font-weight:500; }
        .neighbor-info { flex:1; }
        .neighbor-artist { font-size:12px; color:#8886a8; }
        .neighbor-id { font-size:10px; color:#4a4868; }
        .neighbor-dist { font-size:10px; color:#5a5878; font-variant-numeric:tabular-nums; }

        /* ── Cluster composition ── */
        .artist-cluster-section { padding:14px 18px; border-top:.5px solid #1e1e2c; }
        .artist-cluster-section h4 { font-size:12px; font-weight:500; color:#6a6888; letter-spacing:.08em; text-transform:uppercase; margin:0 0 10px; }
        .artist-stat-row { display:flex; justify-content:space-between; align-items:center; font-size:12px; padding:4px 0; }
        .artist-name { color:#8886a8; }
        .artist-count { display:flex; align-items:center; gap:6px; color:#5a5878; }
        .artist-bar { height:3px; border-radius:2px; background:#3a3560; }

        /* ── Stats bar ── */
        #stats-bar { padding:6px 28px; background:#0a0a10; border-top:.5px solid #1e1e2c; font-size:11px; color:#4a4868; display:flex; gap:24px; }
        #stats-bar span { font-weight:500; color:#6a6888; }

        /* ── Dropdown overrides ── */
        .Select-control { background-color:#1a1a24 !important; border:.5px solid #3a3a50 !important; border-radius:6px !important; }
        .Select-placeholder,.Select-value-label { color:#6a6888 !important; }
        .Select-input input { color:#c8c4e8 !important; }
        .Select-menu-outer { background:#1a1a24 !important; border:.5px solid #3a3a50 !important; }
        .Select-option { color:#c8c4e8 !important; background:#1a1a24 !important; }
        .Select-option:hover,.Select-option.is-focused { background:#2a2a38 !important; }
        .modebar { display:none !important; }
    </style>

    <script>
    // Base64 image map (injectée par Python)
    window.IMAGE_B64_MAP = ''' + IMAGE_B64_JSON + ''';

    // Canvas overlay : bulles dynamiques adaptées au zoom
    document.addEventListener('DOMContentLoaded', function() {

        var canvas   = null;
        var ctx      = null;
        var imgCache = {};      // index -> HTMLImageElement (décodé)
        var hoveredIdx  = null;
        var animFrame   = null;

        // Animation d'agrandissement au survol
        var hoverScale      = 1.0;   // échelle courante de l'animation hover (1 → 3)
        var hoverAnimFrame  = null;
        var hoverTarget     = 3.5;   // facteur d'agrandissement max au survol
        var hoverSpeed      = 0.18;  // vitesse d'interpolation (0-1)

        // Précharger toutes les images dans des objets Image HTML
        function preloadImages() {
            Object.keys(window.IMAGE_B64_MAP).forEach(function(idx) {
                var img = new Image();
                img.src = window.IMAGE_B64_MAP[idx];
                imgCache[idx] = img;
            });
        }
        preloadImages();

        // Attacher le canvas et les listeners dès que Plotly est prêt
        var observer = new MutationObserver(function() {
            var graphDiv = document.getElementById('main-graph');
            if (!graphDiv || graphDiv._canvasAttached) return;

            // Créer le canvas overlay
            var col = document.getElementById('graph-col');
            if (!col) return;
            canvas = document.getElementById('img-canvas');
            if (!canvas) {
                canvas = document.createElement('canvas');
                canvas.id = 'img-canvas';
                canvas.style.position = 'absolute';
                canvas.style.top = '0';
                canvas.style.left = '0';
                canvas.style.pointerEvents = 'none';
                canvas.style.zIndex = '10';
                col.appendChild(canvas);
            }
            ctx = canvas.getContext('2d');
            graphDiv._canvasAttached = true;

            // Redimensionner le canvas quand la fenêtre change
            function resizeCanvas() {
                if (!graphDiv) return;
                canvas.width  = graphDiv.offsetWidth;
                canvas.height = graphDiv.offsetHeight;
                redraw();
            }
            resizeCanvas();
            window.addEventListener('resize', resizeCanvas);

            // Convertir coordonnées data → pixels
            function dataToPixel(gd, x, y) {
                if (!gd._fullLayout || !gd._fullLayout.xaxis || !gd._fullLayout.xaxis._length) return null;
                var ax = gd._fullLayout.xaxis;
                var ay = gd._fullLayout.yaxis;
                var l  = gd._fullLayout.margin.l;
                var t  = gd._fullLayout.margin.t;
                var pw = ax._length;
                var ph = ay._length;
                var px = l + (x - ax.range[0]) / (ax.range[1] - ax.range[0]) * pw;
                var py = t + (1 - (y - ay.range[0]) / (ay.range[1] - ay.range[0])) * ph;
                return {x: px, y: py};
            }

            // Calculer la taille des bulles en fonction du zoom actuel
            // On exprime la taille en unités data puis on la convertit en pixels.
            function getBubbleSizePx(gd) {
                var ax = gd._fullLayout.xaxis;
                var pw = ax._length;
                var dataRange = Math.abs(ax.range[1] - ax.range[0]);
                // 1.2 unités data de diamètre → grossit avec le zoom
                var sizeInPx = (1.2 / dataRange) * pw;
                // Clamp entre 6px (très dézoomé) et 32px (très zoomé)
                return Math.max(6, Math.min(32, sizeInPx));
            }

            // dessine une bulle circulaire avec image (clip avant drawImage)
            function drawBubble(px_x, px_y, size, img, color, alpha, borderColor, borderWidth) {
                var r = size / 2;

                // halo de couleur en fond
                ctx.save();
                ctx.globalAlpha = alpha * 0.25;
                ctx.beginPath();
                ctx.arc(px_x, px_y, r + 1, 0, Math.PI * 2);
                ctx.fillStyle = color || '#7F77DD';
                ctx.fill();
                ctx.restore();

                // image clippée dans le cercle
                ctx.save();
                ctx.beginPath();
                ctx.arc(px_x, px_y, r, 0, Math.PI * 2);
                ctx.closePath();
                ctx.clip();

                if (img && img.complete && img.naturalWidth > 0) {
                    ctx.globalAlpha = alpha;
                    ctx.drawImage(img, px_x - r, px_y - r, size, size);
                } else {
                    ctx.globalAlpha = alpha;
                    ctx.fillStyle = color || '#7F77DD';
                    ctx.fill();
                }
                ctx.restore();

                // bordure circulaire
                if (borderWidth > 0) {
                    ctx.save();
                    ctx.globalAlpha = alpha;
                    ctx.beginPath();
                    ctx.arc(px_x, px_y, r, 0, Math.PI * 2);
                    ctx.strokeStyle = borderColor || 'rgba(255,255,255,0.4)';
                    ctx.lineWidth   = borderWidth;
                    ctx.stroke();
                    ctx.restore();
                }
            }

            // Dessin de toutes les bulles
            function drawAllImages(gd) {
                if (!ctx || !gd || !gd._fullLayout) return;
                ctx.clearRect(0, 0, canvas.width, canvas.height);

                var baseSize = getBubbleSizePx(gd);
                // Images toujours visibles (même petites) — cercle coloré si image absente
                var showBorder = (baseSize >= 8);

                gd.data.forEach(function(trace) {
                    if (!trace.customdata || !trace.x) return;
                    // Récupérer la couleur de la trace (string ou tableau)
                    var traceColor = '#7F77DD';
                    if (trace.marker && trace.marker.color) {
                        traceColor = Array.isArray(trace.marker.color)
                            ? trace.marker.color[0]
                            : trace.marker.color;
                    }

                    for (var i = 0; i < trace.x.length; i++) {
                        var cd = trace.customdata[i];
                        if (!cd) continue;
                        var idx = String(cd[0]);
                        if (idx === String(hoveredIdx)) continue; // dessiné en dernier, agrandi
                        var img = imgCache[idx] || null;
                        var px  = dataToPixel(gd, trace.x[i], trace.y[i]);
                        if (!px) continue;

                        // Couleur par point si disponible (tableau), sinon couleur de trace
                        var ptColor = traceColor;
                        if (Array.isArray(trace.marker && trace.marker.color)) {
                            ptColor = trace.marker.color[i] || traceColor;
                        }

                        drawBubble(
                            px.x, px.y,
                            baseSize,
                            img,
                            ptColor,
                            0.88,
                            showBorder ? 'rgba(255,255,255,0.3)' : 'rgba(0,0,0,0)',
                            showBorder ? 0.8 : 0
                        );
                    }
                });

                // Dessiner le point survolé par-dessus (agrandi + animé)
                if (hoveredIdx !== null) {
                    drawHoveredBubble(gd, baseSize);
                }
            }

            // Dessin animé du point survolé
            function drawHoveredBubble(gd, baseSize) {
                if (hoveredIdx === null || !gd || !gd._fullLayout) return;

                // Taille affichée = base × scale, avec minimum garanti de 60px au hover max
                var scaledSize  = baseSize * hoverScale;
                var displaySize = Math.max(scaledSize, 60 * (hoverScale / hoverTarget));

                gd.data.forEach(function(trace) {
                    if (!trace.customdata || !trace.x) return;
                    for (var i = 0; i < trace.x.length; i++) {
                        var cd = trace.customdata[i];
                        if (!cd) continue;
                        if (String(cd[0]) !== String(hoveredIdx)) continue;

                        var img = imgCache[String(hoveredIdx)] || null;
                        var px  = dataToPixel(gd, trace.x[i], trace.y[i]);
                        if (!px) return;
                        var traceColor = '#7F77DD';
                        if (trace.marker && trace.marker.color) {
                            traceColor = Array.isArray(trace.marker.color)
                                ? (trace.marker.color[i] || trace.marker.color[0])
                                : trace.marker.color;
                        }

                        // Ombre portée
                        ctx.save();
                        ctx.shadowColor   = 'rgba(0,0,0,0.6)';
                        ctx.shadowBlur    = 20;
                        ctx.shadowOffsetY = 5;
                        ctx.beginPath();
                        ctx.arc(px.x, px.y, displaySize / 2 + 3, 0, Math.PI * 2);
                        ctx.fillStyle = 'rgba(255,255,255,0.92)';
                        ctx.fill();
                        ctx.restore();

                        // Bulle image agrandie
                        drawBubble(
                            px.x, px.y,
                            displaySize,
                            img,
                            traceColor,
                            1.0,
                            'rgba(255,255,255,0.95)',
                            2.5
                        );

                        // Anneau lumineux (glow progressif avec l'animation)
                        ctx.save();
                        ctx.beginPath();
                        ctx.arc(px.x, px.y, displaySize / 2 + 5, 0, Math.PI * 2);
                        ctx.strokeStyle = traceColor;
                        ctx.globalAlpha = 0.4 * (hoverScale / hoverTarget);
                        ctx.lineWidth   = 2.5;
                        ctx.stroke();
                        ctx.restore();
                    }
                });
            }

            // Animation smooth du zoom au survol
            function animateHover(targetScale) {
                if (hoverAnimFrame) cancelAnimationFrame(hoverAnimFrame);
                function step() {
                    hoverScale += (targetScale - hoverScale) * hoverSpeed;
                    if (Math.abs(hoverScale - targetScale) < 0.01) {
                        hoverScale = targetScale;
                    }
                    redraw();
                    if (Math.abs(hoverScale - targetScale) > 0.01) {
                        hoverAnimFrame = requestAnimationFrame(step);
                    }
                }
                step();
            }

            function redraw() {
                if (animFrame) cancelAnimationFrame(animFrame);
                animFrame = requestAnimationFrame(function() {
                    drawAllImages(graphDiv);
                });
            }

            // Events Plotly
            graphDiv.on('plotly_hover', function(data) {
                var pt = data.points[0];
                if (!pt.customdata) return;
                hoveredIdx = pt.customdata[0];
                hoverScale = 1.0;
                animateHover(hoverTarget);
            });

            graphDiv.on('plotly_unhover', function() {
                hoveredIdx = null;
                animateHover(1.0);
            });

            // Redessiner après chaque relayout (zoom/pan) → taille des bulles recalculée
            graphDiv.on('plotly_relayout', function() {
                resizeCanvas();
            });

            // Redessiner après react (nouvelle figure) — avec retry si _fullLayout pas encore prêt
            graphDiv.on('plotly_react', function() {
                resizeCanvas();
            });
            graphDiv.on('plotly_afterplot', function() {
                resizeCanvas();
            });

            // Premier dessin robuste : retry jusqu'à ce que _fullLayout soit disponible
            function tryFirstDraw(attempts) {
                if (attempts <= 0) return;
                if (graphDiv._fullLayout && graphDiv._fullLayout.xaxis && graphDiv._fullLayout.xaxis._length) {
                    resizeCanvas();
                } else {
                    setTimeout(function() { tryFirstDraw(attempts - 1); }, 150);
                }
            }
            tryFirstDraw(20);
        });

        observer.observe(document.body, { childList:true, subtree:true });
    });
    </script>
</head>
<body>
    {%app_entry%}
    <footer>
        {%config%}
        {%scripts%}
        {%renderer%}
    </footer>
</body>
</html>
'''

# layout
app.layout = html.Div([

    html.Div([
        html.H1("AbstractViz"),
        html.Span(f"{n_items} œuvres · {len(ARTISTS)} artistes · {len(CLUSTERS)} classes")
    ], id='header'),

    html.Div([
        html.Label("Chercher une œuvre :"),
        dcc.Dropdown(
            id='search-bar',
            options=dropdown_options,
            placeholder="Index ou artiste…",
            clearable=True,
            searchable=True,
            style={'flex':1, 'maxWidth':'420px'}
        ),
        html.Div([
            html.Label("Couleur :"),
            dcc.RadioItems(
                id='color-mode',
                options=[
                    {'label': 'Par classe', 'value': 'cluster'},
                    {'label': 'Par artiste', 'value': 'artist'}
                ],
                value='cluster',
                inline=True,
                style={'fontSize':'12px','color':'#8886a8','gap':'12px'}
            )
        ], style={'display':'flex','alignItems':'center','gap':'10px','marginLeft':'auto'})
    ], id='search-wrapper'),

    html.Div([
        html.Label("Filtrer par artiste :"),
        dcc.Dropdown(
            id='artist-filter-dropdown',
            options=[{'label': a, 'value': a} for a in ARTISTS],
            placeholder="Tous les artistes…",
            clearable=True,
            searchable=True,
            style={'flex':1,'maxWidth':'300px'}
        ),
        html.Button("✕ Réinitialiser", id='btn-all-artists', className='artist-btn', n_clicks=0,
                    style={'marginLeft':'8px'})
    ], id='filter-wrapper'),

    html.Div(id='mode-info', className='mode-info'),

    html.Div([

        html.Div([
            dcc.Graph(
                id='main-graph',
                style={'height':'100%','width':'100%'},
                clear_on_unhover=True,
                config={'scrollZoom':True,'displayModeBar':False,'doubleClick':'reset'}
            )
        ], id='graph-col'),

        html.Div(id='detail-panel', children=[
            html.Div([
                html.Div("◎", className='icon'),
                html.P("Survolez un point pour un aperçu rapide.\nCliquez pour explorer l'œuvre en détail.")
            ], className='panel-placeholder')
        ])

    ], id='main-layout'),

    html.Div(id='stats-bar'),

    dcc.Store(id='selected-point', data=None),
    dcc.Store(id='active-artist', data=None),
    dcc.Store(id='current-artwork', data=None),
])


# callback : graphe
@app.callback(
    Output('main-graph', 'figure'),
    Output('stats-bar', 'children'),
    Output('mode-info', 'children'),
    Output('mode-info', 'className'),
    Input('selected-point', 'data'),
    Input('color-mode', 'value'),
    Input('active-artist', 'data'),
)
def update_graph(selected_index, color_mode, active_artist):
    fig = go.Figure()

    filtered_df = df[df['artist'] == active_artist].copy() if active_artist else df.copy()
    background_df = df[~df.index.isin(filtered_df.index)] if active_artist else pd.DataFrame()

    if len(background_df) > 0:
        fig.add_trace(go.Scattergl(
            x=background_df['x'], y=background_df['y'],
            mode='markers', name='_bg',
            marker=dict(size=10, color='#1e1e2c', opacity=0.0, line=dict(width=0)),
            hoverinfo='skip', showlegend=False
        ))

    def add_points(sub, name, color):
        # customdata[3] = image_path pour le JS hover thumbnail
        cd = np.stack([sub['index'], sub['artist'], sub['cluster'], sub['image_path']], axis=-1)
        fig.add_trace(go.Scattergl(
            x=sub['x'], y=sub['y'],
            mode='markers', name=name,
            # Marqueurs transparents : hitbox active pour hover/clic,
            # le rendu visuel est entièrement géré par le canvas overlay
            marker=dict(size=14, color=color, opacity=0.0, line=dict(width=0)),
            customdata=cd,
            hovertemplate="<b>#%{customdata[0]}</b> — %{customdata[1]}<br>Classe %{customdata[2]}<extra></extra>"
        ))

    if color_mode == 'cluster':
        for cid in sorted(filtered_df['cluster'].unique()):
            add_points(filtered_df[filtered_df['cluster'] == cid], f"Classe {cid}", get_cluster_color(cid))
    else:
        for artist in ARTISTS:
            sub = filtered_df[filtered_df['artist'] == artist]
            if len(sub): add_points(sub, artist, get_artist_color(artist))

    if selected_index is not None:
        sel = df[df['index'] == selected_index].iloc[0]
        neighbors = compute_nearest_neighbors(selected_index, n=8)
        neighbor_rows = df[df['index'].isin([n['index'] for n in neighbors])]

        for _, nr in neighbor_rows.iterrows():
            fig.add_trace(go.Scattergl(
                x=[sel['x'], nr['x'], None], y=[sel['y'], nr['y'], None],
                mode='lines', line=dict(color='rgba(127,119,221,0.2)', width=1),
                hoverinfo='skip', showlegend=False
            ))

        fig.add_trace(go.Scattergl(
            x=neighbor_rows['x'], y=neighbor_rows['y'],
            mode='markers', name='Voisins',
            marker=dict(size=18, color='rgba(127,119,221,0.0)', line=dict(width=2, color='#7F77DD')),
            hoverinfo='skip', showlegend=False
        ))

        fig.add_trace(go.Scattergl(
            x=[sel['x']], y=[sel['y']],
            mode='markers', name='Sélection',
            marker=dict(size=22, color='rgba(0,0,0,0)',
                        line=dict(width=3, color='#ffffff')),
            hoverinfo='skip', showlegend=False
        ))

        same = df[(df['artist']==sel['artist']) & (df['cluster']==sel['cluster']) & (df['index']!=selected_index)]
        if len(same):
            fig.add_trace(go.Scattergl(
                x=same['x'], y=same['y'],
                mode='markers', name=f"{sel['artist']} — même classe",
                marker=dict(size=8, color=get_artist_color(sel['artist']),
                            symbol='diamond', line=dict(width=0.5, color='#ffffff')),
                hoverinfo='skip', showlegend=False
            ))

    fig.update_layout(
        margin=dict(l=0,r=0,b=0,t=0),
        hovermode='closest',
        plot_bgcolor='#0f0f13', paper_bgcolor='#0f0f13',
        xaxis=dict(showgrid=False,zeroline=False,showticklabels=False,showline=False),
        yaxis=dict(showgrid=False,zeroline=False,showticklabels=False,showline=False,scaleanchor='x'),
        legend=dict(font=dict(size=11,color='#8886a8',family='Inter'),
                    bgcolor='rgba(15,15,19,0.8)',borderwidth=0,x=0.01,y=0.99),
        dragmode='pan', uirevision='constant'
    )

    n_visible = len(filtered_df)
    stats = [f"{n_visible} œuvres  ·  {filtered_df['cluster'].nunique()} classes  ·  molette=zoom  ·  clic=détail  ·  double-clic=reset"]
    mode_text = [f"◉  Filtré : {active_artist} ({n_visible} œuvres)"] if active_artist else []
    mode_class = 'mode-info visible' if active_artist else 'mode-info'

    return fig, stats, mode_text, mode_class


# callback : panneau détail
@app.callback(
    Output('detail-panel', 'children'),
    Output('selected-point', 'data'),
    Output('current-artwork', 'data'),
    Input('main-graph', 'clickData'),
    Input('search-bar', 'value'),
    State('selected-point', 'data')
)
def update_detail_panel(clickData, searched_index, current_selected):
    ctx = callback_context
    triggered = ctx.triggered[0]['prop_id'] if ctx.triggered else ''

    target_index = None
    if 'search-bar' in triggered and searched_index is not None:
        target_index = searched_index
    elif 'main-graph' in triggered and clickData:
        pt = clickData['points'][0]
        if 'customdata' in pt:
            target_index = int(pt['customdata'][0])

    if target_index is None:
        return [html.Div([
            html.Div("◎", className='icon'),
            html.P("Survolez un point pour l'aperçu.\nCliquez pour le détail complet.")
        ], className='panel-placeholder')], None, dash.no_update

    row = df[df['index'] == target_index]
    if len(row) == 0:
        return dash.no_update, dash.no_update, dash.no_update
    row = row.iloc[0]

    cluster_id    = int(row['cluster'])
    artist        = row['artist']
    cluster_color = get_cluster_color(cluster_id)
    neighbors     = compute_nearest_neighbors(target_index, n=6)
    artist_in_cluster = compute_artist_in_cluster(cluster_id)
    max_count     = max(c for _,c,_ in artist_in_cluster) if artist_in_cluster else 1
    same_ac       = df[(df['artist']==artist) & (df['cluster']==cluster_id)]
    same_total    = df[df['artist']==artist]
    pct           = round(len(same_ac)/max(len(same_total),1)*100,1)

    panel_content = [

        # ── Image ──
        html.Div([
            html.Img(src=f"assets/abstrait-v4/{row['image_path']}",
                     style={'maxWidth':'100%','maxHeight':'240px','objectFit':'contain','borderRadius':'4px'})
        ], className='detail-img-wrapper'),

        # ── Méta ──
        html.Div([
            html.H3(f"Œuvre #{target_index}"),
            html.Div([html.Span("Artiste",className='meta-label'), html.Span(artist,className='meta-value')], className='meta-row'),
            html.Div([
                html.Span("Classe",className='meta-label'),
                html.Div([html.Div(className='cluster-dot',style={'background':cluster_color}),
                          html.Span(f"Classe {cluster_id}",className='meta-value')],
                         style={'display':'flex','alignItems':'center','gap':'5px'})
            ], className='meta-row'),
            html.Div([html.Span(f"{artist} dans cette classe",className='meta-label'),
                      html.Span(f"{len(same_ac)} œuvres ({pct}%)",className='meta-value')], className='meta-row'),
            html.Div([html.Span(f"{artist} hors classe",className='meta-label'),
                      html.Span(f"{len(same_total)-len(same_ac)} œuvres",className='meta-value')], className='meta-row'),
        ], className='detail-meta'),

        # ── Résumé IA Gemini ──
        html.Div([
            html.H4(["Analyse de l'œuvre ", html.Span("Groq AI", className='ai-badge')]),
            html.Div(id='ai-content', children=[
                html.Button("✦ Générer l'analyse IA", id='ai-btn', className='ai-btn', n_clicks=0)
            ])
        ], className='ai-section'),

        # ── Voisins ──
        html.Div([
            html.H4("Voisins les plus proches"),
            *[html.Div([
                html.Span(f"{i+1}", className='neighbor-rank'),
                html.Div([
                    html.Div(n['artist'], className='neighbor-artist'),
                    html.Div(f"#{n['index']} — Classe {n['cluster']}", className='neighbor-id'),
                ], className='neighbor-info'),
                html.Span(f"d={n['distance']:.2f}", className='neighbor-dist'),
            ], className='neighbor-item') for i,n in enumerate(neighbors)]
        ], className='neighbors-section'),

        # ── Composition du cluster ──
        html.Div([
            html.H4(f"Composition de la classe {cluster_id}"),
            *[html.Div([
                html.Span(a, className='artist-name',
                          style={'fontWeight':'500' if a==artist else '400',
                                 'color': get_artist_color(a) if a==artist else '#8886a8'}),
                html.Div([
                    html.Div(style={'width':f"{int(c/max_count*80)}px",'height':'3px',
                                    'borderRadius':'2px',
                                    'background':get_artist_color(a) if a==artist else '#3a3a50'},
                             className='artist-bar'),
                    html.Span(f"{p}%", className='artist-count')
                ], style={'display':'flex','alignItems':'center','gap':'6px'})
            ], className='artist-stat-row') for a,c,p in artist_in_cluster]
        ], className='artist-cluster-section'),
    ]

    artwork_data = {'artist': artist, 'image_path': row['image_path']}
    return panel_content, target_index, artwork_data


# callback : appel Groq
@app.callback(
    Output('ai-content', 'children'),
    Input('ai-btn', 'n_clicks'),
    State('current-artwork', 'data'),
    prevent_initial_call=True
)
def generate_ai_analysis(n_clicks, artwork_data):
    if not n_clicks or not artwork_data:
        return dash.no_update

    artist   = artwork_data.get('artist', '')
    img_path = artwork_data.get('image_path', '')

    result = call_groq(artist, img_path)

    return html.P(result, className='ai-text')


# callback : filtre artiste
@app.callback(
    Output('active-artist', 'data'),
    Input('artist-filter-dropdown', 'value'),
    Input('btn-all-artists', 'n_clicks'),
    prevent_initial_call=True
)
def filter_by_artist(dropdown_value, reset_clicks):
    ctx = callback_context
    if not ctx.triggered:
        return None
    if 'btn-all-artists' in ctx.triggered[0]['prop_id']:
        return None
    return dropdown_value


if __name__ == '__main__':
    print(f"{n_items} œuvres | {len(ARTISTS)} artistes | {len(CLUSTERS)} classes")
    print(f"Clé Groq : {'configurée' if GROQ_API_KEY != 'VOTRE_CLE_GROQ_ICI' else 'à renseigner (ligne 12)'}")
    app.run(debug=True)