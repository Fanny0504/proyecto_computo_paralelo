"""Prueba de humo del nucleo: compila, compara seq/par/blas y revisa la calidad."""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pca_numba as p


def imagen_sintetica(h=128, w=192):
    y, x = np.mgrid[0:h, 0:w]
    r = 127 + 100 * np.sin(x / 20.0) * np.cos(y / 35.0)
    g = 127 + 100 * np.sin((x + y) / 40.0)
    b = 127 + 100 * np.cos(x / 30.0 - y / 50.0)
    return np.stack([r, g, b], axis=-1).clip(0, 255)


def main():
    X = imagen_sintetica()

    t = time.perf_counter()
    p.warmup()
    print(f"warm-up (compilacion): {time.perf_counter() - t:.1f} s")

    k = 8
    rec_s, stored, evr = p.compress_channels(X, k, "seq")
    rec_p, _, _ = p.compress_channels(X, k, "par")
    rec_b, _, _ = p.compress_channels(X, k, "blas")

    d_sp = np.abs(rec_s - rec_p).max()
    d_sb = np.abs(rec_s - rec_b).max()
    calidad = p.psnr(X, rec_s)
    ratio = X.size / stored

    print(f"max|seq - par|  = {d_sp:.2e}")
    print(f"max|seq - blas| = {d_sb:.2e}")
    print(f"PSNR = {calidad:.1f} dB | ratio = {ratio:.2f}:1 | varianza explicada = {evr:.4f}")

    assert d_sp < 1e-5, "seq y par no coinciden"
    assert d_sb < 1e-3, "seq y blas no coinciden"
    assert calidad > 40, "la calidad es demasiado baja"
    print("OK: todas las comprobaciones pasaron")


if __name__ == "__main__":
    main()