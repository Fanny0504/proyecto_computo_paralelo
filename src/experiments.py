import os
import sys
import time
import csv
import numpy as np
from PIL import Image
from numba import set_num_threads, get_num_threads

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import src.pca_numba as pca_numba

def ejecutar_pca(img, k=32):
    # Llama a la función exacta de Fanny activando su modo paralelo
    res = pca_numba.compress_channels(img, k, 'par')
    return res[0] if isinstance(res, tuple) else res

def main():
    print("Iniciando pruebas de PCA con src/pca_numba.py...")
    ruta_img = "data/dataset_fanny/kodim01.png"
    if os.path.exists(ruta_img):
        img = np.asarray(Image.open(ruta_img).convert("RGB"))
        print(f"Imagen cargada: {ruta_img}")
    else:
        img = np.random.randint(0, 256, (1024, 1024, 3), dtype=np.uint8)
        print("Imagen aleatoria (1024x1024).")

    # Calentamiento
    set_num_threads(1)
    _ = ejecutar_pca(img[:64, :64], k=5)

    max_hilos = get_num_threads()
    
    # 1. CORRECTITUD
    set_num_threads(1)
    res_1 = ejecutar_pca(img, k=32)
    set_num_threads(max_hilos)
    res_p = ejecutar_pca(img, k=32)

    diff = np.max(np.abs(res_1.astype(int) - res_p.astype(int)))
    print(f"\n=== 1. Correctitud (1 vs {max_hilos} hilos) ===")
    print(f"Diferencia máxima: {diff} px -> {'OK' if diff <= 1 else 'Revisar'}\n")

    # 2. ESCALABILIDAD
    print("=== 2. Escalabilidad por Hilos ===")
    print(f"{'Hilos':>6} | {'Tiempo (s)':>10} | {'Speedup':>9} | {'Eficiencia':>10}")
    print("-" * 45)

    hilos_prueba = [h for h in [1, 2, 4, 8, 16] if h <= max_hilos]
    t_base = None
    filas_csv = []

    for h in hilos_prueba:
        set_num_threads(h)
        t0 = time.perf_counter()
        _ = ejecutar_pca(img, k=32)
        t_actual = time.perf_counter() - t0

        if t_base is None: t_base = t_actual
        speedup = t_base / t_actual
        eficiencia = speedup / h
        filas_csv.append([h, round(t_actual, 4), round(speedup, 2), round(eficiencia, 2)])
        print(f"{h:>6} | {t_actual:>10.4f} | {speedup:>8.2f}x | {eficiencia*100:>9.1f}%")

    os.makedirs("results", exist_ok=True)
    with open("results/escalabilidad.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["threads", "time_s", "speedup", "efficiency"])
        writer.writerows(filas_csv)
    print("\n[OK] Resultados guardados en results/escalabilidad.csv")

if __name__ == "__main__":
    main()