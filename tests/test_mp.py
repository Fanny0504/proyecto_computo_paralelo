"""Prueba de pca_mp: compara varios procesos y los dos modos contra una referencia directa."""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pca_mp as p


def imagen_sintetica(h=128, w=192):
    y, x = np.mgrid[0:h, 0:w]
    r = 127 + 100 * np.sin(x / 20.0) * np.cos(y / 35.0)
    g = 127 + 100 * np.sin((x + y) / 40.0)
    b = 127 + 100 * np.cos(x / 30.0 - y / 50.0)
    img = np.stack([r, g, b], axis=-1)
    ruido = np.random.default_rng(0).normal(0, 3, img.shape)
    return (img + ruido).clip(0, 255)


def referencia(X_rgb, k):
    """PCA directo con NumPy, sin partir en bloques ni usar procesos."""
    rec = np.empty_like(X_rgb)
    for c in range(3):
        X = X_rgb[:, :, c]
        mu = X.mean(axis=0)
        vals, vecs = np.linalg.eigh(np.cov(X, rowvar=False))
        Vk = vecs[:, ::-1][:, :k]
        rec[:, :, c] = np.clip((X - mu) @ Vk @ Vk.T + mu, 0, 255)
    return rec


def main():
    X = imagen_sintetica()
    k = 8
    ref = referencia(X, k)

    configuraciones = [("filas", 1), ("filas", 2), ("filas", 4), ("canales", 2), ("canales", 3)]
    peor = 0.0
    for modo, procesos in configuraciones:
        t = time.perf_counter()
        with p.PCAParalelo(procesos, modo) as pca:
            t_arranque = time.perf_counter() - t
            t = time.perf_counter()
            rec, stored, evr = pca.comprimir(X, k)
            t_calculo = time.perf_counter() - t
            etapas = pca.etapas
        dif = float(np.abs(rec - ref).max())
        peor = max(peor, dif)
        print(f"modo={modo:8s} procesos={procesos}: arranque={t_arranque:.2f} s | calculo={t_calculo:.3f} s | "
              f"max|rec - ref| = {dif:.2e} | etapas: " + ", ".join(f"{n}={v:.3f}" for n, v in etapas.items()))

    calidad = p.psnr(X, rec)
    print(f"PSNR = {calidad:.1f} dB | ratio = {X.size / stored:.2f}:1 | varianza explicada = {evr:.4f}")

    assert peor < 1e-6, "alguna configuracion no coincide con la referencia"
    assert calidad > 30, "la calidad es demasiado baja"
    print("OK: todas las comprobaciones pasaron")


if __name__ == "__main__":
    main()
