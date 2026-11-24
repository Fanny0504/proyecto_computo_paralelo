"""Perfil por etapas y comparacion de modos de paralelismo (filas / canales) de pca_mp."""
import argparse
import csv
import os
import time

import numpy as np
from PIL import Image

from pca_mp import PCAParalelo


def mediana(valores):
    return float(np.median(valores))


def medir(X, k, p, modo, reps):
    with PCAParalelo(p, modo) as pca:
        pca.comprimir(X, k)  # calentamiento (no se mide)
        totales, etapas = [], []
        for _ in range(reps):
            t = time.perf_counter()
            rec, _, _ = pca.comprimir(X, k)
            totales.append(time.perf_counter() - t)
            etapas.append(dict(pca.etapas))
    por_etapa = {e: mediana([d[e] for d in etapas]) for e in etapas[0]}
    return rec, mediana(totales), por_etapa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--procs", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--modos", nargs="+", default=["filas", "canales"], choices=["filas", "canales"])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--tag", required=True, help="nombre de la maquina, por ejemplo fanny")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    X = np.asarray(Image.open(a.input).convert("RGB"), dtype=np.float64)
    h, w, _ = X.shape
    nombre = os.path.splitext(os.path.basename(a.input))[0]
    print(f"Imagen {nombre}: {w}x{h} | k={a.k} | nucleos logicos: {os.cpu_count()} | maquina: {a.tag}")
    print("Nota: en modo 'canales' los tiempos por etapa son la suma del trabajo de cada canal (se solapan en el tiempo real).")

    # base: version secuencial (modo filas con 1 proceso)
    ref, t1, et1 = medir(X, a.k, 1, "filas", a.reps)
    otros1 = t1 - sum(et1.values())
    fr_secuencial = (et1["eigh"] + max(otros1, 0.0)) / t1
    print(f"\nPerfil secuencial (p=1): T={t1:.3f} s | covarianza={et1['covarianza']:.3f} s "
          f"({et1['covarianza'] / t1:.0%}) | eigh={et1['eigh']:.3f} s ({et1['eigh'] / t1:.0%}) | "
          f"proyeccion={et1['proyeccion']:.3f} s ({et1['proyeccion'] / t1:.0%}) | otros={otros1:.3f} s")
    if fr_secuencial < 1:
        print(f"Fraccion secuencial estimada (eigh + otros) = {fr_secuencial:.0%} -> "
              f"speedup maximo teorico (Amdahl) = {1 / fr_secuencial:.2f}\n")

    filas = [("filas", 1, t1, 1.0, et1, otros1, 0.0)]
    for modo in a.modos:
        for p in a.procs:
            if p == 1:
                continue
            rec, t, et = medir(X, a.k, p, modo, a.reps)
            dif = float(np.abs(rec - ref).max())
            otros = t - sum(et.values()) if modo == "filas" else float("nan")
            filas.append((modo, p, t, t1 / t, et, otros, dif))
            print(f"modo={modo:8s} p={p:2d} | T={t:8.3f} s | speedup={t1 / t:5.2f} | eficiencia={t1 / t / p:4.2f} | "
                  f"cov={et['covarianza']:.3f} eigh={et['eigh']:.3f} proy={et['proyeccion']:.3f} | dif={dif:.1e}")

    salida = os.path.join(a.out, f"perfil_{a.tag}_{nombre}.csv")
    with open(salida, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["maquina", "imagen", "ancho", "alto", "k", "modo", "procesos", "tiempo_s", "speedup",
                     "eficiencia", "t_covarianza", "t_eigh", "t_proyeccion", "dif_max_vs_secuencial"])
        for modo, p, t, s, et, otros, dif in filas:
            wr.writerow([a.tag, nombre, w, h, a.k, modo, p, round(t, 4), round(s, 3), round(s / p, 3),
                         round(et["covarianza"], 4), round(et["eigh"], 4), round(et["proyeccion"], 4), f"{dif:.2e}"])
    print(f"Resultados en {salida}")


if __name__ == "__main__":
    main()
