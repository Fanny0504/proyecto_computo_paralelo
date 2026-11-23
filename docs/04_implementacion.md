# Implementación con `multiprocessing`

## Arquitectura de la clase `PCAParalelo`

La paralelización se encapsuló en la clase `PCAParalelo` dentro de `src/pca_mp.py`, diseñada como un administrador de contexto (`with PCAParalelo(procesos=p) as pca:`). Al entrar al bloque `with`:

- Si `procesos == 1`, no se inicializa ningún grupo de procesos y las tareas se ejecutan de forma directa en el proceso principal.
- Si `procesos > 1`, se crea **una única instancia** de `multiprocessing.Pool(processes=p)` que se reutiliza a lo largo de los tres canales (R, G y B) y de todas las invocaciones a `pca.comprimir(X_rgb, k)` dentro del contexto. Al salir del bloque `with`, el `Pool` se cierra y libera sus recursos ordenadamente.

Este diseño evita el error común de crear y destruir procesos en cada canal o en cada repetición del experimento.

## Arranque con `spawn` en Windows

A diferencia de Linux (que tradicionalmente utiliza `fork`), en Windows el método por defecto de `multiprocessing` es **`spawn`**. Esto implica que cada proceso trabajador inicia un intérprete de Python limpio, vuelve a importar el módulo y carga las librerías necesarias (NumPy y `threadpoolctl`) antes de quedar listo para recibir tareas.

En las mediciones realizadas en Windows, este costo inicial de creación del `Pool` oscila entre **1.86 s y 2.72 s** dependiendo del número de procesos. Dado que es un costo fijo de inicialización independiente de cuántas imágenes se compriman, el script `src/experiments.py` mide el tiempo de arranque por separado y cronometra únicamente `pca.comprimir(X, k)` tras una ejecución de calentamiento no contabilizada.

## Funciones a nivel de módulo y serialización (`pickle`)

Bajo el modelo `spawn` en Windows, tanto la función a ejecutar como los argumentos y los valores de retorno deben serializarse mediante `pickle` para viajar a través de tuberías (*pipes*) de comunicación entre procesos (IPC). Por este motivo:

- Las funciones trabajadoras (como `_estadisticos` y la función de proyección/reconstrucción por bloque) están definidas a **nivel de módulo** en `src/pca_mp.py`, en lugar de ser funciones anidadas o métodos ligados que no pueden serializarse directamente con `pickle`.
- Los datos se pasan como arreglos contiguos de NumPy (`np.ascontiguousarray`) en `float64`.

## Verificación de equivalencia numérica

En `src/experiments.py`, cada ejecución compara la imagen reconstruida con $p$ procesos frente a la reconstrucción secuencial ($p=1$) calculando $\max |\hat{X}_p - \hat{X}_1|$. En todas las combinaciones evaluadas sobre los cuatro mosaicos Kodak la diferencia máxima registrada fue **`0.00e+00`**, confirmando que la descomposición por filas y la reducción de estadísticos preservan la exactitud numérica respecto a la ejecución secuencial.