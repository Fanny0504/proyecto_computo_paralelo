"""Calidad de la compresion PCA sobre una carpeta de imagenes (por defecto, Kodak)."""
import argparse
import csv
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from pca_numba import compress_channels, psnr, mse

EXTENSIONES = ("*.png", "*.bmp", "*.jpg", "*.jpeg")


def listar_imagenes(carpeta):
    archivos = []
    for e in EXTENSIONES:
        archivos += glob.glob(os.path.join(carpeta, e))
    return sorted(archivos)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/dataset_fanny")
    ap.add_argument("--out", default="results")
    ap.add_argument("--tag", default="kodak", help="prefijo de los archivos de salida")
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 2, 5, 10, 20, 30, 50, 75])
    ap.add_argument("--mode", default="blas", choices=["seq", "par", "blas"])
    ap.add_argument("--limit", type=int, default=0, help="usar solo las primeras N imagenes (0 = todas)")
    ap.add_argument("--example", default=None, help="nombre de archivo para la figura visual")
    ap.add_argument("--show-ks", type=int, nargs="+", default=[5, 20, 50])
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    files = listar_imagenes(a.data)
    if a.limit > 0:
        files = files[:a.limit]
    if not files:
        raise SystemExit(f"No hay imagenes en {a.data}")

    rows = []
    for path in files:
        name = os.path.basename(path)
        X = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64)
        h, w, _ = X.shape
        original = h * w * 3
        for k in a.ks:
            if k > w:
                continue
            rec, stored, evr = compress_channels(X, k, a.mode)
            rows.append({"imagen": name, "k": k, "ratio": original / stored,
                         "psnr_db": psnr(X, rec), "mse": mse(X, rec), "var_explicada": evr})
        print("listo", name)

    # --- CSV por imagen ---
    with open(os.path.join(a.out, f"{a.tag}_por_imagen.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    # --- resumen por k ---
    ks = sorted({r["k"] for r in rows})
    summary = []
    for k in ks:
        sel = [r for r in rows if r["k"] == k]
        p = np.array([r["psnr_db"] for r in sel])
        summary.append({"k": k, "n_imagenes": len(sel),
                        "ratio_prom": np.mean([r["ratio"] for r in sel]),
                        "psnr_prom": p.mean(), "psnr_std": p.std(),
                        "psnr_min": p.min(), "psnr_max": p.max(),
                        "var_expl_prom": np.mean([r["var_explicada"] for r in sel])})
    with open(os.path.join(a.out, f"{a.tag}_resumen.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        wr.writeheader()
        wr.writerows({k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()} for s in summary)

    # --- mejor y peor imagen con un k de referencia ---
    kref = 20 if 20 in ks else ks[len(ks) // 2]
    ref = sorted((r for r in rows if r["k"] == kref), key=lambda r: r["psnr_db"])
    print(f"\nk={kref}: peor {ref[0]['imagen']} ({ref[0]['psnr_db']:.2f} dB) | "
          f"mejor {ref[-1]['imagen']} ({ref[-1]['psnr_db']:.2f} dB)")

    # --- graficas ---
    kk = [s["k"] for s in summary]
    pm = np.array([s["psnr_prom"] for s in summary])
    ps = np.array([s["psnr_std"] for s in summary])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(kk, pm, "o-")
    ax[0].fill_between(kk, pm - ps, pm + ps, alpha=0.25)
    ax[0].set(xlabel="k (componentes)", ylabel="PSNR promedio (dB)", title="Calidad vs k (+/- 1 desv. est.)")
    ax[1].plot([s["ratio_prom"] for s in summary], pm, "o-")
    for s in summary:
        ax[1].annotate(f"k={s['k']}", (s["ratio_prom"], s["psnr_prom"]), fontsize=7)
    ax[1].set(xlabel="razon de compresion (x:1)", ylabel="PSNR promedio (dB)", title="Calidad vs compresion")
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, f"{a.tag}_calidad.png"), dpi=150)

    # --- figura visual de una imagen ---
    ex = next((f for f in files if a.example and os.path.basename(f) == a.example), files[0])
    X = np.asarray(Image.open(ex).convert("RGB"), dtype=np.float64)
    h, w, _ = X.shape
    fig, ax = plt.subplots(1, 1 + len(a.show_ks), figsize=(4 * (1 + len(a.show_ks)), 3.5))
    ax[0].imshow(X.astype(np.uint8))
    ax[0].set_title("Original")
    ax[0].axis("off")
    for i, k in enumerate(a.show_ks, start=1):
        rec, stored, _ = compress_channels(X, k, a.mode)
        ax[i].imshow(rec.astype(np.uint8))
        ax[i].set_title(f"k={k} | {h * w * 3 / stored:.1f}:1 | {psnr(X, rec):.1f} dB", fontsize=9)
        ax[i].axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, f"{a.tag}_ejemplo.png"), dpi=150)
    print(f"Figuras y CSV en {a.out}/")


if __name__ == "__main__":
    main()