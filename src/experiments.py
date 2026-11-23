"""Escalabilidad de pca_mp: tiempo, speedup y eficiencia segun el numero de procesos."""
import argparse
import csv
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from pca_mp import PCAParalelo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="imagen o mosaico, por ejemplo data/scaled/kodak_mosaico_4x3.png")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--procs", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--tag", required=True, help="nombre de la maquina, por ejemplo kurt o fanny")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    X = np.asarray(Image.open(a.input).convert("RGB"), dtype=np.float64)
    h, w, _ = X.shape
    nombre = os.path.splitext(os.path.basename(a.input))[0]
    print(f"Imagen {nombre}: {w}x{h} | k={a.k} | nucleos logicos: {os.cpu_count()} | maquina: {a.tag}")

    filas, t1, ref = [], None, None
    for p in a.procs:
        if p > (os.cpu_count() or 1):
            print(f"aviso: p={p} supera los nucleos logicos ({os.cpu_count()})")
        t0 = time.perf_counter()
        with PCAParalelo(p) as pca:
            arranque = time.perf_counter() - t0
            rec, _, _ = pca.comprimir(X, a.k)  # calentamiento (no se mide)
            tiempos = []
            for _ in range(a.reps):
                t = time.perf_counter()
                pca.comprimir(X, a.k)
                tiempos.append(time.perf_counter() - t)
        mediana = float(np.median(tiempos))
        if ref is None:
            ref = rec
        dif = float(np.abs(rec - ref).max())
        t1 = t1 or mediana
        speedup = t1 / mediana
        filas.append((p, arranque, mediana, speedup, speedup / p, dif))
        print(f"p={p:2d} | arranque={arranque:5.2f} s | T={mediana:8.3f} s | speedup={speedup:5.2f} | "
              f"eficiencia={speedup / p:4.2f} | max|rec - rec(p=1)|={dif:.1e}")

    base = os.path.join(a.out, f"escalabilidad_{a.tag}_{nombre}")
    with open(base + ".csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["maquina", "imagen", "ancho", "alto", "k", "procesos", "arranque_s",
                     "tiempo_s", "speedup", "eficiencia", "dif_max_vs_p1"])
        for p, ar, t, s, e, d in filas:
            wr.writerow([a.tag, nombre, w, h, a.k, p, round(ar, 3), round(t, 4), round(s, 3), round(e, 3), f"{d:.2e}"])

    ps = [r[0] for r in filas]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(ps, [r[3] for r in filas], "o-", label="real")
    ax[0].plot(ps, ps, "--", label="ideal")
    ax[0].set(xlabel="procesos", ylabel="speedup", title=f"Speedup ({a.tag}, {w}x{h})")
    ax[0].legend()
    ax[1].plot(ps, [r[4] for r in filas], "o-")
    ax[1].set(xlabel="procesos", ylabel="eficiencia", title="Eficiencia", ylim=(0, 1.1))
    fig.tight_layout()
    fig.savefig(base + ".png", dpi=150)
    print(f"Resultados en {base}.csv y {base}.png")


if __name__ == "__main__":
    main()