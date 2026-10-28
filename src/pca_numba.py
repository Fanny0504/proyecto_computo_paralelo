"""PCA por canal con kernels Numba: secuencial, paralelo y baseline BLAS."""
import numpy as np
from numba import njit, prange


@njit(cache=True)
def matmul_seq(A, B):
    N, M = A.shape
    P = B.shape[1]
    C = np.zeros((N, P), dtype=np.float64)
    for i in range(N):
        for k in range(M):          # orden i,k,j: acceso contiguo a B y C
            a = A[i, k]
            for j in range(P):
                C[i, j] += a * B[k, j]
    return C


@njit(parallel=True, cache=True)
def matmul_par(A, B):
    N, M = A.shape
    P = B.shape[1]
    C = np.zeros((N, P), dtype=np.float64)
    for i in prange(N):             # cada hilo escribe filas distintas
        for k in range(M):
            a = A[i, k]
            for j in range(P):
                C[i, j] += a * B[k, j]
    return C


MODES = {"seq": matmul_seq, "par": matmul_par, "blas": np.dot}


def pca_fit_transform(X, k, mode):
    mm = MODES[mode]
    mean = X.mean(axis=0)
    Xc = np.ascontiguousarray(X - mean)
    cov = mm(np.ascontiguousarray(Xc.T), Xc) / (X.shape[0] - 1)
    vals, vecs = np.linalg.eigh(cov)             # orden ascendente
    vals = vals[::-1]
    Vk = np.ascontiguousarray(vecs[:, ::-1][:, :k])
    Z = mm(Xc, Vk)
    return Z, Vk, mean, vals


def pca_reconstruct(Z, Vk, mean, mode):
    mm = MODES[mode]
    return np.clip(mm(Z, np.ascontiguousarray(Vk.T)) + mean, 0, 255)


def compress_channels(X_rgb, k, mode):
    """X_rgb: HxWx3 float64. Devuelve (reconstruida, bytes_teoricos, varianza_explicada)."""
    h, w, c = X_rgb.shape
    rec = np.empty_like(X_rgb)
    evr = []
    for ch in range(c):
        Z, Vk, mean, vals = pca_fit_transform(X_rgb[:, :, ch], k, mode)
        rec[:, :, ch] = pca_reconstruct(Z, Vk, mean, mode)
        evr.append(vals[:k].sum() / vals.sum())
    stored = c * (h * k + w * k + w) * 4         # float32
    return rec, stored, float(np.mean(evr))


def warmup():
    """Compila con los mismos tipos y layouts que se usaran al medir."""
    X = np.random.rand(32, 32, 3) * 255
    for mode in ("seq", "par"):
        compress_channels(X, 4, mode)


def mse(a, b):
    return float(np.mean((a - b) ** 2))


def psnr(a, b):
    m = mse(a, b)
    return float("inf") if m == 0 else 10 * np.log10(255.0 ** 2 / m)