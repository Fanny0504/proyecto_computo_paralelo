"""Arma mosaicos de imagenes Kodak de distintos tamanos para las pruebas de rendimiento."""
import argparse
import glob
import os

from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/dataset_fanny")
    ap.add_argument("--outdir", default="data/scaled")
    ap.add_argument("--grids", nargs="+", default=["1x1", "2x2", "4x3", "6x4"], help="formato COLUMNASxFILAS")
    a = ap.parse_args()

    archivos = sorted(glob.glob(os.path.join(a.data, "*.png")))
    if not archivos:
        raise SystemExit(f"No hay imagenes PNG en {a.data}")

    tiles = []
    for f in archivos:
        im = Image.open(f).convert("RGB")
        if im.height > im.width:  # las verticales se giran para que todas midan igual
            im = im.transpose(Image.ROTATE_90)
        tiles.append(im)
    tw, th = tiles[0].size
    tiles = [t if t.size == (tw, th) else t.resize((tw, th)) for t in tiles]

    os.makedirs(a.outdir, exist_ok=True)
    for g in a.grids:
        cols, filas = (int(x) for x in g.split("x"))
        if cols * filas > len(tiles):
            print(f"se omite {g}: faltan imagenes ({len(tiles)} disponibles)")
            continue
        lienzo = Image.new("RGB", (cols * tw, filas * th))
        for i in range(cols * filas):
            lienzo.paste(tiles[i], ((i % cols) * tw, (i // cols) * th))
        salida = os.path.join(a.outdir, f"kodak_mosaico_{g}.png")
        lienzo.save(salida)
        print(f"{salida}  ({lienzo.width}x{lienzo.height})")


if __name__ == "__main__":
    main()