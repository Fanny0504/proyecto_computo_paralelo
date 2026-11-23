# Discusión y análisis según la ley de Amdahl

## Límite teórico de aceleración

La ley de Amdahl establece que el *speedup* alcanzable con $p$ procesos está acotado por la fracción secuencial $s$ del programa:

$$S(p) = \frac{1}{s + \frac{1 - s}{p}}$$

En las mediciones de la máquina de Kurt el *speedup* máximo observado fue **1.060×** (`6x4`, $p=2$) y **1.028×** (`2x2`, $p=4$), muy lejos del escalado lineal ideal $S(p) = p$. Este comportamiento responde a cinco factores concretos que interactúan en la implementación actual:

## 1. Dominancia de `np.linalg.eigh` como cuello de botella secuencial

La etapa paralela de covarianza tiene complejidad $O(h w^2)$ y las de proyección/reconstrucción $O(h w k)$, mientras que la descomposición espectral `np.linalg.eigh` sobre la matriz de covarianza $C \in \mathbb{R}^{w \times w}$ es secuencial y escala como $O(w^3)$.

En los cuatro mosaicos evaluados el ancho $w$ es siempre mayor que el alto $h$:
- En `1x1` y `2x2`, $w = 1.5\,h$.
- En `4x3`, $w = 2.0\,h$ ($3072 \times 1536$).
- En `6x4`, $w = 2.25\,h$ ($4608 \times 2048$).

Al cumplirse $w > h$, el término secuencial $O(w^3)$ crece más rápido que el término paralelizable $O(h w^2)$. De hecho, al pasar de `2x2` ($w=1536$, $T(1)=4.24\text{ s}$) a `6x4` ($w=4608$, tres veces más ancho), el tiempo secuencial aumenta **21.2 veces** ($89.65\text{ s}$), un crecimiento cercano al cubo de $w$ ($3^3 = 27$). Esto hace que la fracción secuencial $s$ domine el tiempo total y fije un techo muy bajo en la ley de Amdahl.

## 2. Costo de comunicación entre procesos (`pickle`) y saturación en $p=16$

En la Etapa 1, cada uno de los $p$ procesos calcula y devuelve al proceso principal una matriz completa $G_i = X_i^T X_i$ de tamaño $w \times w$ en `float64`. A diferencia de los bloques de filas (cuyo tamaño individual disminuye al dividir entre $p$), **el tamaño de $G_i$ no disminuye al aumentar $p$**, por lo que el volumen total de datos serializados y copiados por *pipes* crece como $O(p \cdot w^2)$ por canal:

- En `4x3` ($w=3072$), cada $G_i$ ocupa ~72 MiB (~576 MiB transferidos en $p=8$ por canal).
- En `6x4` ($w=4608$), cada $G_i$ ocupa ~162 MiB (~1.27 GiB transferidos en $p=8$ por canal).

Este crecimiento lineal con $p$ explica tanto por qué el rendimiento empeora al subir de $p=2$ o $p=4$ a $p=8$, como el fallo `OSError 1450` observado en Windows al intentar transferir simultáneamente 16 matrices parciales (~1.13 GiB en `4x3` y ~2.53 GiB en `6x4` por canal).

## 3. Reducción secuencial y ancho de banda de memoria

Una vez recibidas las $p$ matrices $G_i$, el proceso principal debe sumarlas secuencialmente ($O(p w^2)$), lo que añade trabajo secuencial proporcional a $p$. Adicionalmente, los productos matriciales y las copias de arreglos grandes compiten por el mismo ancho de banda de la memoria RAM y las cachés compartidas L3, reduciendo la ganancia efectiva conforme más procesos acceden a memoria simultáneamente.

## 4. Costo de arranque del `Pool` (`spawn` en Windows)

Aunque el arranque del `Pool` se midió fuera de `T(p)`, su magnitud (1.86–2.72 s) demuestra que en Windows paralelizar imágenes individuales pequeñas no resulta rentable si el `Pool` no se amortiza procesando un lote grande de imágenes dentro de un mismo contexto `with`.

## 5. Núcleos físicos frente a hilos lógicos

El equipo evaluado cuenta con 8 núcleos físicos y 16 hilos lógicos (SMT). En los mosaicos donde $p=16$ logró completarse (`1x1` y `2x2`), duplicar los procesos de 8 a 16 empeoró notablemente los tiempos (`0.805 s` $\rightarrow$ `1.029 s` en `1x1` y `4.492 s` $\rightarrow$ `5.536 s` en `2x2`). En cargas intensivas de punto flotante (`float64`), los hilos lógicos adicionales compiten por las mismas unidades aritméticas de los 8 núcleos físicos, aumentan el cambio de contexto y duplican el tráfico de `pickle`, degradando la eficiencia por debajo del 5 %.

## Por qué el desempeño relativo mejora ligeramente al crecer la imagen

En `1x1` (0.64 s), la latencia fija de despachar tareas por canal en `pool.map` y serializar las matrices domina por completo al cómputo, haciendo que $S(p) < 1$ en todos los puntos. Al escalar a `2x2` y `6x4`, la carga aritmética de $X_i^T X_i$, proyección y reconstrucción es suficientemente grande como para compensar esa latencia inicial con pocos procesos ($p=2$ o $p=4$), logrando superar ligeramente la barrera de $1.0\times$ (`1.028×` en `2x2` y `1.060×` en `6x4`). Sin embargo, a partir de $p \ge 4$ el crecimiento del tráfico IPC ($O(p w^2)$) y el peso de `eigh` ($O(w^3)$) vuelven a cancelar cualquier ganancia adicional.