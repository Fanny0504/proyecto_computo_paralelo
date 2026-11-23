# Diseño paralelo

## Descomposición por bloques de filas

El algoritmo procesa cada canal de color (R, G y B) de forma independiente como una matriz $X \in \mathbb{R}^{h \times w}$ en precisión `float64`. Para distribuir el trabajo entre $p$ procesos sin dependencias de escritura, la matriz del canal se divide horizontalmente en $p$ bloques contiguos de filas ($X_1, X_2, \dots, X_p$), asignando un bloque a cada proceso trabajador.

Cuando `procesos=1`, el algoritmo ejecuta la misma lógica directamente en el proceso principal sin crear un `Pool`, sirviendo como línea base secuencial exacta.

## Etapas del algoritmo por canal

El procesamiento de cada canal se divide en tres etapas claramente diferenciadas:

1. **Etapa 1 — Estadísticos parciales y reducción de covarianza (paralela + reducción secuencial):**
   - **Cálculo en paralelo:** cada proceso recibe su bloque de filas $X_i$ (con $n_i$ filas) y calcula de forma independiente tres estadísticos locales: el número de filas $n_i$, la suma por columnas $s_i = \sum_{r \in X_i} x_r$ y el producto cruzado parcial $G_i = X_i^T X_i \in \mathbb{R}^{w \times w}$.
   - **Reducción en el proceso principal:** al recibir los resultados parciales de los $p$ procesos, el proceso principal consolida el total de filas $h = \sum_i n_i$, la suma global $S = \sum_i s_i$ y la matriz de productos cruzados acumulada $G = \sum_i G_i$. Con estos acumulados reconstruye la media por columna y la matriz de covarianza sin requerir una segunda lectura paralela sobre $X$:

     $$\mu = \frac{1}{h} S, \qquad C = \frac{1}{h - 1}\left(G - h\,\mu^T \mu\right)$$

2. **Etapa 2 — Descomposición espectral (secuencial):**
   - El proceso principal calcula los autovalores y autovectores de la matriz simétrica $C \in \mathbb{R}^{w \times w}$ mediante `np.linalg.eigh`, ordena las componentes de mayor a menor varianza y selecciona las primeras $k$ columnas para formar $V_k \in \mathbb{R}^{w \times k}$.
   - Esta etapa tiene complejidad $O(w^3)$ y no se paraleliza entre procesos, por lo que constituye la principal fracción secuencial que acota el *speedup* según la ley de Amdahl.

3. **Etapa 3 — Proyección y reconstrucción (paralela):**
   - Cada proceso recibe su bloque de filas $X_i$, el vector de medias $\mu$ y la matriz de proyección $V_k$.
   - De forma independiente centra su bloque ($X_{c,i} = X_i - \mu$), proyecta al subespacio reducido ($Z_i = X_{c,i} V_k$) y reconstruye su porción del canal ($\hat{X}_i = Z_i V_k^T + \mu$). El proceso principal concatena los bloques reconstruidos y recorta los valores al intervalo $[0, 255]$.

## Control de hilos en BLAS (`threadpoolctl`)

Internamente, operaciones de NumPy como `@` y `np.linalg.eigh` delegan en librerías BLAS/LAPACK multihilo (por ejemplo, OpenBLAS o MKL). Si cada uno de los $p$ procesos de `multiprocessing` lanzara a su vez múltiples hilos BLAS, se produciría **sobresuscripción de hilos** (contención por núcleos físicos y saturación de caché), además de que el caso base `p=1` ya estaría usando múltiples núcleos de forma oculta.

Por esta razón, el módulo restringe BLAS a **1 hilo por proceso** mediante `threadpoolctl`. De este modo, `p=1` representa un tiempo secuencial genuino en un solo hilo y cualquier aceleración observada con $p > 1$ proviene exclusivamente del paralelismo explícito entre procesos.