# Compresión de imágenes con PCA en cómputo paralelo

Proyecto de cómputo paralelo: compresión de imágenes con PCA (Análisis de Componentes Principales), con una versión secuencial y una versión paralela con `multiprocessing` de Python. Se evalúan la calidad de la compresión sobre el conjunto Kodak y el desempeño de la paralelización (tiempo, speedup, eficiencia y perfil por etapas).

## Método

PCA se aplica a cada canal de color (R, G, B). En cada canal, cada fila de píxeles es una muestra y cada columna una característica: se calcula la covarianza (ancho × ancho), su descomposición espectral con `np.linalg.eigh`, y se conservan los `k` autovectores principales. La imagen comprimida se guarda como `Z` (alto × k), la base `Vk` (ancho × k) y la media, en `float32`.

La paralelización (`src/pca_mp.py`, clase `PCAParalelo`) usa un `multiprocessing.Pool` creado una sola vez y BLAS limitado a un hilo, de modo que la aceleración proviene solo de los procesos. Hay dos modos:

- **`filas`**: cada canal se parte en bloques de filas. La covarianza (con reducción de estadísticos parciales) y la proyección/reconstrucción corren en paralelo; `eigh` es secuencial.
- **`canales`**: cada proceso calcula el PCA completo de un canal, incluido `eigh` (máximo 3 procesos).

## Resultados principales (medidos)

**Calidad** (promedio de las 24 imágenes de Kodak; la razón de compresión se calcula contra la imagen sin comprimir):

| k | Razón de compresión | PSNR promedio (dB) |
|---|---|---|
| 5 | 13.8:1 | 21.3 |
| 20 | 3.7:1 | 24.9 |
| 50 | 1.5:1 | 28.1 |

La tabla completa está en `results/kodak_resumen.csv` y las figuras en `results/kodak_calidad.png` y `results/kodak_ejemplo.png`.

**Rendimiento** (k = 20, mosaicos de imágenes Kodak):

- En el modo `filas` no hay aceleración útil en ninguna de las dos máquinas (speedup máximo medido: 1.06).
- El perfil por etapas (mosaico 2x2, `p = 1`) muestra que `eigh`, que es secuencial, toma el 84 % del tiempo (covarianza 11 %, proyección 3 %), lo que limita el speedup a ≈ 1.16 según la ley de Amdahl.
- El modo `canales` llega a un speedup de 1.84 con 3 procesos (mosaico 2x2, máquina de Fanny).
- Detalles en `results/escalabilidad_*.csv` y `results/perfil_*.csv`.

## Estructura del repositorio

```
src/
  pca_mp.py          PCA con multiprocessing (clase PCAParalelo, modos filas y canales)
  quality_dataset.py evaluación de calidad sobre una carpeta de imágenes
  build_mosaic.py    arma mosaicos grandes con imágenes de Kodak
  experiments.py     escalabilidad: tiempo, speedup y eficiencia según el número de procesos
  perfil_etapas.py   perfil por etapas y comparación de los modos filas y canales
  pca_numba.py       primera versión con Numba (solo referencia, no se usa en el proyecto)
tests/
  test_mp.py         correctitud de pca_mp contra una referencia de NumPy
  test_nucleo.py     prueba de la versión inicial con Numba (referencia)
data/dataset_fanny/  conjunto Kodak (24 imágenes PNG)
results/             CSV y gráficas de los experimentos
docs/                reporte y datos de hardware
download_data.py     descarga de un conjunto de datos más grande (opcional)
```

Los archivos `pca_paralelo.py` y `prueba.py` de la raíz son material previo del equipo.

## Instalación

Requiere Python 3.10 o superior.

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Cómo reproducir

Ejecutar desde la raíz del repositorio con el entorno activado:

```
python tests\test_mp.py
python src\quality_dataset.py --example kodim23.png --procs 4
python src\build_mosaic.py
python src\experiments.py --input data\scaled\kodak_mosaico_2x2.png --k 20 --procs 1 2 4 8 --reps 3 --tag mi_maquina
python src\perfil_etapas.py --input data\scaled\kodak_mosaico_2x2.png --k 20 --procs 1 2 3 4 --reps 2 --tag mi_maquina
```

Los mosaicos se generan en `data/scaled/`, que no se versiona. Los resultados se guardan en `results/`. Durante las mediciones conviene no usar la computadora para no distorsionar los tiempos.

## Notas

- En Windows los procesos se crean con `spawn`, lo que añade cerca de un segundo de arranque; se mide aparte del tiempo de cálculo.
- Con imágenes grandes y 16 procesos en el modo `filas`, Windows reportó `OSError 1450` (recursos insuficientes): cada proceso devuelve su matriz de covarianza parcial (ancho × ancho) por un pipe.
- Con imágenes de píxeles enteros, las sumas parciales son exactas y el resultado es idéntico para cualquier número de procesos.

## Equipo

- Estefanny Sanluis Carcaño (Fanny0504)
- Kurt (Kurtmnoyola)
