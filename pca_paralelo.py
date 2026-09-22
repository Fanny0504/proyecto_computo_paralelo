import os
import time
import numpy as np
from numba import njit, prange
import matplotlib.pyplot as plt
from PIL import Image

# ==========================================
# 1. NÚCLEOS MATEMÁTICOS (Secuencial vs Paralelo)
# ==========================================

@njit
def multiplicacion_matrices_secuencial(A, B):
    """VERSIÓN 1: Un solo núcleo (Baseline para medir Speedup)"""
    N = A.shape[0]
    P = B.shape[1]
    M = A.shape[1]
    
    C = np.zeros((N, P), dtype=np.float32)
    for i in range(N):
        for j in range(P):
            suma = 0.0
            for k in range(M):
                suma += A[i, k] * B[k, j]
            C[i, j] = suma
    return C

@njit(parallel=True)
def multiplicacion_matrices_paralela(A, B):
    """VERSIÓN 2: Múltiples núcleos (Tu implementación paralela)"""
    N = A.shape[0]
    P = B.shape[1]
    M = A.shape[1]
    
    C = np.zeros((N, P), dtype=np.float32)
    for i in prange(N): 
        for j in range(P):
            suma = 0.0
            for k in range(M):
                suma += A[i, k] * B[k, j]
            C[i, j] = suma
    return C

# ==========================================
# 2. ALGORITMO PCA 
# ==========================================

def pca_comprimir(X, num_componentes, usar_paralelo):
    media = np.mean(X, axis=0)
    X_centrada = X - media
    X_transpuesta = X_centrada.T
    
    if usar_paralelo:
        matriz_covarianza = multiplicacion_matrices_paralela(X_transpuesta, X_centrada) / (X.shape[0] - 1)
    else:
        matriz_covarianza = multiplicacion_matrices_secuencial(X_transpuesta, X_centrada) / (X.shape[0] - 1)
    
    eigen_valores, eigen_vectores = np.linalg.eigh(matriz_covarianza)
    indices_ordenados = np.argsort(eigen_valores)[::-1]
    eigen_vectores_ordenados = eigen_vectores[:, indices_ordenados]
    
    V_k = eigen_vectores_ordenados[:, :num_componentes].astype(np.float32)
    
    if usar_paralelo:
        Z = multiplicacion_matrices_paralela(X_centrada.astype(np.float32), V_k)
    else:
        Z = multiplicacion_matrices_secuencial(X_centrada.astype(np.float32), V_k)
    
    return Z, V_k, media

def pca_restaurar(Z, V_k, media, usar_paralelo):
    V_k_transpuesta = V_k.T
    
    if usar_paralelo:
        X_reconstruida_centrada = multiplicacion_matrices_paralela(Z, V_k_transpuesta)
    else:
        X_reconstruida_centrada = multiplicacion_matrices_secuencial(Z, V_k_transpuesta)
    
    X_final = X_reconstruida_centrada + media
    X_final = np.clip(X_final, 0, 255)
    
    return X_final

# ==========================================
# 3. FLUJO PRINCIPAL 
# ==========================================

def procesar_imagen_color():
    # Asegúrate de poner aquí el nombre correcto (ej. 'prueba.bmp' si estás usando ese formato)
    ruta_imagen = 'prueba3.bmp' 
    
    try:
        img = Image.open(ruta_imagen).convert('RGB')
    except FileNotFoundError:
        print(f"ERROR: No se encontró '{ruta_imagen}'.")
        return

    X_original_rgb = np.array(img, dtype=np.float32)
    alto, ancho, canales = X_original_rgb.shape
    num_componentes = 20 
    
    print(f"\n--- PROYECTO: PCA PARALELO ---")
    print(f"Resolución original: {ancho}x{alto} px")
    print(f"Componentes conservados por canal: {num_componentes} (de {ancho})")
    
    print("\n[INFO] Compilando funciones Numba en C (Warm-up)...")
    dummy = np.ones((10, 10), dtype=np.float32)
    _ = multiplicacion_matrices_secuencial(dummy, dummy)
    _ = multiplicacion_matrices_paralela(dummy, dummy)
    print("[INFO] Compilación lista.\n")

    # EJECUCIÓN 1: SECUENCIAL (1 NÚCLEO)
    print(">>> EJECUTANDO VERSIÓN SECUENCIAL (1 Hilo) ...")
    inicio_sec = time.time()
    for canal in range(3):
        X_canal = X_original_rgb[:, :, canal]
        Z, Vk, M = pca_comprimir(X_canal, num_componentes, usar_paralelo=False)
        _ = pca_restaurar(Z, Vk, M, usar_paralelo=False)
    fin_sec = time.time()
    tiempo_secuencial = fin_sec - inicio_sec
    print(f"Tiempo Secuencial (T1): {tiempo_secuencial:.2f} segundos\n")

    # EJECUCIÓN 2: PARALELA (MULTINÚCLEO)
    print(">>> EJECUTANDO VERSIÓN PARALELA (Multinúcleo) ...")
    inicio_par = time.time()
    Z_list, V_list, media_list = [], [], []
    for canal in range(3):
        X_canal = X_original_rgb[:, :, canal]
        Z, Vk, M = pca_comprimir(X_canal, num_componentes, usar_paralelo=True)
        Z_list.append(Z)
        V_list.append(Vk)
        media_list.append(M)
        
    X_restaurada_rgb = np.zeros_like(X_original_rgb)
    for canal in range(3):
        X_restaurada_rgb[:, :, canal] = pca_restaurar(Z_list[canal], V_list[canal], media_list[canal], usar_paralelo=True)
    
    fin_par = time.time()
    tiempo_paralelo = fin_par - inicio_par
    print(f"Tiempo Paralelo (Tp): {tiempo_paralelo:.2f} segundos\n")

    # ---------------------------------------------------------
    # CÁLCULO DE MÉTRICAS 
    # ---------------------------------------------------------
    speedup = tiempo_secuencial / tiempo_paralelo
    eficiencia = speedup / os.cpu_count() * 100 

    print(f"=====================================")
    print(f"RESULTADOS ACADÉMICOS PARA EL REPORTE:")
    print(f"=====================================")
    print(f"Speedup obtenido: {speedup:.2f}x")
    print(f"Núcleos lógicos detectados: {os.cpu_count()}")
    print(f"Eficiencia aproximada: {eficiencia:.2f}%")
    print(f"-------------------------------------")
    
    # GUARDAR PARA MEDIR EL TAMAÑO FÍSICO
    archivo_comprimido = 'vectores_pca.npz'
    np.savez_compressed(archivo_comprimido, 
                        Z0=Z_list[0], V0=V_list[0], M0=media_list[0],
                        Z1=Z_list[1], V1=V_list[1], M1=media_list[1],
                        Z2=Z_list[2], V2=V_list[2], M2=media_list[2])
    
    peso_foto_original = os.path.getsize(ruta_imagen) / 1024
    peso_vectores_pca = os.path.getsize(archivo_comprimido) / 1024
    
    print(f"Peso foto original: {peso_foto_original:.2f} KB")
    print(f"Peso vectores PCA: {peso_vectores_pca:.2f} KB")
    print(f"Tamaño reducido al: {(peso_vectores_pca/peso_foto_original)*100:.2f}%")
    print(f"=====================================\n")
    
    # --- VISUALIZACIÓN ---
    X_original_rgb = X_original_rgb.astype(np.uint8)
    X_restaurada_rgb = X_restaurada_rgb.astype(np.uint8)
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title(f"Original ({peso_foto_original:.0f} KB)")
    plt.imshow(X_original_rgb)
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title(f"Restaurada - {num_componentes} Componentes ({peso_vectores_pca:.0f} KB)")
    plt.imshow(X_restaurada_rgb)
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    procesar_imagen_color()