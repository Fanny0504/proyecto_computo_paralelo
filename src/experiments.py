import os
import sys
import glob
import time
import csv
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.pca_mp import PCAParalelo


def cargar_mosaico_kodak(grid=(4, 6), tile_size=(768, 512)):
    ruta_mosaico = "data/mosaics/mosaic_kodak_4608x2048.png"
    if os.path.exists(ruta_mosaico):
        return np.asarray(Image.open(ruta_mosaico).convert("RGB"))

    archivos = sorted(glob.glob("data/dataset_fanny/*.png")) or sorted(glob.glob("data/kodak/*.png"))
    if not archivos:
        raise FileNotFoundError("No se encontraron imagenes Kodak en data/dataset_fanny/ ni en data/kodak/.")

    rows, cols = grid
    tw, th = tile_size
    mosaico = Image.new("RGB", (cols * tw, rows * th))

    for idx in range(rows * cols):
        with Image.open(archivos[idx % len(archivos)]) as img:
            tile = img.convert("RGB").resize((tw, th), Image.Resampling.LANCZOS)
            mosaico.paste(tile, ((idx % cols) * tw, (idx // cols) * th))

    os.makedirs(os.path.dirname(ruta_mosaico), exist_ok=True)
    mosaico.save(ruta_mosaico)
    return np.asarray(mosaico)


def main():
    X = cargar_mosaico_kodak()
    k = 30
    repeticiones = 3
    procesos_lista = [1, 2, 4, 8, 16]

    resultados = []
    t_secuencial = None
    rec_base = None

    print(f"Mosaico Kodak: {X.shape[1]}x{X.shape[0]} px | k={k} | repeticiones={repeticiones}")
    print(f"{'p':>4} | {'Pool (s)':>10} | {'Mediana PCA (s)':>15} | {'Speedup':>9} | {'Eficiencia':>10} | {'Max Diff':>8}")
    print("-" * 70)

    for p in procesos_lista:
        t0_pool = time.perf_counter()
        with PCAParalelo(procesos=p) as pca:
            t_pool = time.perf_counter() - t0_pool

            tiempos = []
            for _ in range(repeticiones):
                t0 = time.perf_counter()
                rec, n_bytes, var = pca.comprimir(X, k)
                tiempos.append(time.perf_counter() - t0)

        t_med = float(np.median(tiempos))

        if p == 1:
            t_secuencial = t_med
            rec_base = rec

        diff_max = int(np.max(np.abs(rec_base.astype(np.int32) - rec.astype(np.int32))))
        speedup = t_secuencial / t_med
        eficiencia = speedup / p

        resultados.append([
            p,
            round(t_pool, 6),
            round(t_med, 6),
            round(speedup, 4),
            round(eficiencia, 4),
            diff_max,
        ])

        print(f"{p:>4} | {t_pool:>10.4f} | {t_med:>15.4f} | {speedup:>8.2f}x | {eficiencia*100:>9.1f}% | {diff_max:>8d}")

    os.makedirs("results", exist_ok=True)
    ruta_csv = "results/escalabilidad_hp_pavilion_ryzen7.csv"
    with open(ruta_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["procesos", "tiempo_pool_s", "tiempo_pca_mediana_s", "speedup", "eficiencia", "max_diff_px"])
        writer.writerows(resultados)

    print(f"\nGuardado en: {ruta_csv}")


if __name__ == "__main__":
    main()