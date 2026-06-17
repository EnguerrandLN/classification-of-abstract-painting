import os
from PIL import Image

# Tes dossiers
base_folder = r"C:\projet im06\serieux"
output_folder = r"C:\projet im06\serieux_compresse"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

print("⏳ Recherche et compression des images...")

compteur = 0
# os.walk permet d'explorer tous les sous-dossiers automatiquement
for root, dirs, files in os.walk(base_folder):
    # On évite de scanner le dossier de sortie s'il est à l'intérieur
    if output_folder in root:
        continue
        
    for filename in files:
        # On ignore le script Python lui-même et les fichiers de config
        if filename.endswith('.py') or filename.endswith('.csv') or filename.startswith('.'):
            continue
            
        img_path = os.path.join(root, filename)
        
        try:
            # On demande à PIL d'essayer d'ouvrir le fichier (ça plantera gentiment si ce n'est pas une image)
            img = Image.open(img_path)
            
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")
            
            # On s'assure que le fichier de sortie sera bien un .jpg
            nom_sans_ext = os.path.splitext(filename)[0]
            out_path = os.path.join(output_folder, f"{nom_sans_ext}.jpg")

            img.thumbnail((800, 800), Image.Resampling.LANCZOS)
            img.save(out_path, "JPEG", quality=80, optimize=True)
            
            compteur += 1
            if compteur % 100 == 0:
                print(f"✓ {compteur} images compressées...")

        except Exception:
            # Ce fichier n'était pas une image valide, on passe au suivant
            pass

print(f"\n✅ Terminé ! {compteur} images traitées.")
print(r"▶ Va vérifier le dossier 'C:\projet im06\serieux_compresse' !")