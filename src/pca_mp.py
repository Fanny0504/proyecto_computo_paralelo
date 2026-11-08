"""PCA por canal con multiprocessing: version secuencial (1 proceso) y paralela (Pool)."""
import os

# BLAS de un solo hilo: asi el speedup viene solo de los procesos.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

from multiprocessing import Pool

import numpy as np

try:
    from threadpoolctl import threadpool_limits
except ImportError:  # sin threadpoolctl se depende de las variables de entorno
    threadpool_limits = None

_LIMITE = None


def _limitar_blas():
    """Limita BLAS a 1 hilo en este proceso (se usa tambien como initializer del Pool)."""
    global _LIMITE
    if threadpool_limits is not None:
        _LIMITE = threadpool_limits(limits=1)


_limitar_blas()


# ---------- tareas que se ejecutan en los procesos (deben estar a nivel de modulo) ----------
def _estadisticos(chunk):
    """Etapa 1: estadisticos parciales de un bloque de filas."""
    c = chunk.astype(np.float64, copy=False)
    return c.shape[0], c.sum(axis=0), c.T @ c


def _proyectar_reconstruir(args):
    """Etapa 3: proyecta y reconstruye un bloque de filas."""
    chunk, mean, Vk = args
    Z = (chunk - mean) @ Vk
    rec = np.clip(Z @ Vk.T + mean, 0, 255)
    return Z, rec


def _noop(_):
    return 0


# ---------- clase principal ----------
class PCAParalelo:
    """Comprime imagenes con PCA. procesos=1 es la version secuencial; >1 usa un Pool.

    Uso:
        with PCAParalelo(procesos=4) as pca:
            rec, bytes_almacenados, var_explicada = pca.comprimir(X_rgb, k)
    """

    def __init__(self, procesos=1):
        self.p = max(1, int(procesos))
        self.pool = None

    def __enter__(self):
        if self.p > 1:
            self.pool = Pool(self.p, initializer=_limitar_blas)
            self.pool.map(_noop, range(self.p * 4), chunksize=1)  # calentamiento: arranca los workers
        return self

    def __exit__(self, *exc):
        if self.pool is not None:
            self.pool.close()
            self.pool.join()
            self.pool = None

    def _mapa(self, fn, items):
        if self.pool is None:
            return [fn(i) for i in items]
        return self.pool.map(fn, items)

    def comprimir_canal(self, canal, k):
        chunks = np.array_split(canal, self.p, axis=0)  # bloques de filas

        # Etapa 1 (paralela): estadisticos parciales y reduccion
        parciales = self._mapa(_estadisticos, chunks)
        n = sum(s[0] for s in parciales)
        suma = sum(s[1] for s in parciales)
        gram = sum(s[2] for s in parciales)
        mean = suma / n
        cov = (gram - n * np.outer(mean, mean)) / (n - 1)

        # Etapa 2 (secuencial): descomposicion espectral
        vals, vecs = np.linalg.eigh(cov)
        vals = vals[::-1]
        Vk = np.ascontiguousarray(vecs[:, ::-1][:, :k])

        # Etapa 3 (paralela): proyeccion y reconstruccion
        res = self._mapa(_proyectar_reconstruir, [(c, mean, Vk) for c in chunks])
        Z = np.vstack([r[0] for r in res])
        rec = np.vstack([r[1] for r in res])
        return rec, Z, Vk, mean, vals

    def comprimir(self, X_rgb, k):
        """X_rgb: HxWx3 float64. Devuelve (reconstruida, bytes_teoricos, varianza_explicada)."""
        h, w, c = X_rgb.shape
        rec = np.empty_like(X_rgb)
        evr = []
        for ch in range(c):
            r, Z, Vk, mean, vals = self.comprimir_canal(np.ascontiguousarray(X_rgb[:, :, ch]), k)
            rec[:, :, ch] = r
            evr.append(vals[:k].sum() / vals.sum())
        stored = c * (h * k + w * k + w) * 4  # float32
        return rec, stored, float(np.mean(evr))


# ---------- metricas ----------
def mse(a, b):
    return float(np.mean((a - b) ** 2))


def psnr(a, b):
    m = mse(a, b)
    return float("inf") if m == 0 else 10 * np.log10(255.0 ** 2 / m)