# Trabajo futuro

A partir de los cuellos de botella identificados en los experimentos con `multiprocessing`, se proponen tres líneas de mejora concretas para escalar tanto la velocidad como la eficiencia de memoria:

## 1. Uso de memoria compartida (`multiprocessing.shared_memory`)

El mayor freno técnico en mosaicos grandes (incluyendo el error `OSError 1450` en $p=16$ para `4x3` y `6x4`) es la copia y serialización de matrices densas a través de *pipes* mediante `pickle`. Sustituir el paso de mensajes por bloques de `multiprocessing.shared_memory` permitiría:

- Alojar la imagen de entrada y la matriz reconstruida en memoria compartida de cero copias (*zero-copy*), enviando a cada proceso únicamente índices de inicio y fin de fila.
- Acumular las matrices parciales de covarianza de forma controlada sin duplicar gigabytes de datos en los búferes de comunicación de Windows.

## 2. PCA por bloques de 8 × 8 píxeles

El enfoque actual aplica PCA globalmente sobre filas de ancho $w$, lo que genera una matriz de covarianza de dimensión $w \times w$ y un costo secuencial $O(w^3)$ en `np.linalg.eigh` que se vuelve prohibitivo en resoluciones altas ($w = 4608$). Dividir cada canal en bloques locales de $8 \times 8$ píxeles (aplanados a vectores de 64 elementos):

- Fija el tamaño de la matriz de covarianza en **$64 \times 64$ independientemente de la resolución de la imagen**, reduciendo el costo de `eigh` a una fracción despreciable y disminuyendo drásticamente la fracción secuencial $s$ de la ley de Amdahl.
- Hace que la relación entre filas (número de bloques) y columnas (64) sea altamente favorable al paralelismo por bloques de filas.
- Permite además comparar directamente la calidad tasa-distorsión frente a estándares por bloques como JPEG.

## 3. Comparación sistemática frente a Numba en memoria compartida

El repositorio conserva una implementación inicial con Numba (`src/pca_numba.py`) basada en hilos dentro de un solo proceso (`@njit(parallel=True)` y `prange`). Como continuación natural, conviene ejecutar el mismo protocolo de mosaicos (`1x1` a `6x4`) comparando ambos paradigmas en igualdad de condiciones:

- **Procesos (`multiprocessing`):** memoria aislada, sin GIL, pero con penalización de `spawn` y `pickle`.
- **Hilos nativos (`Numba`):** memoria compartida real sin serialización entre hilos, con costo único de compilación JIT, permitiendo cuantificar con precisión cuánto del *overhead* medido en este proyecto pertenece al modelo de procesos en Windows y cuánto al límite intrínseco de `eigh`.