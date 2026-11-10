# TP2 – Perceptrón Multicapa (MLP) para clasificación de texto en 20 Newsgroups

Trabajo práctico de Redes Neuronales. Extiende la **regresión softmax** a un **MLP** (capas ocultas + no linealidades) sobre vectores TF-IDF del dataset *20 Newsgroups* (~18.000 posts de Usenet, 20 categorías). El MLP y el modelo lineal se comparan bajo exactamente el mismo protocolo: mismo preprocesamiento, mismo espacio de búsqueda, mismo número de trials, mismas semillas y mismo hardware.

> Este notebook también genera el archivo `overfit_config.json` que se usa como punto de partida en el TP3 (regularización). Ver la sección [Relación con el TP3](#relación-con-el-tp3).

## Contenido del repositorio

| Archivo | Descripción |
|---|---|
| `TP2_MLP.ipynb` | Notebook con todo el trabajo (código, experimentos, análisis y conclusiones). |
| `requirements.txt` | Dependencias de Python (las mismas que usa el TP3). |
| `README.md` | Este archivo. |
| `results/` | Resultados en CSV del notebook (grilla de overfitting, trials y pruning de los estudios, resúmenes por hiperparámetro, ablaciones, resultados finales por semilla, etc.), las bases de los estudios de Optuna (`optuna_*.db`) y `overfit_config.json`. |
| `runs/` | Logs de TensorBoard de cada entrenamiento. Se visualizan con `tensorboard --logdir runs`. |
| `mlflow.db` | Base SQLite de MLflow con el experimento `mlp_20newsgroups`: hiperparámetros, métricas por época y tiempos. Se explora con `mlflow ui --backend-store-uri sqlite:///mlflow.db`. |
| `mlruns/` | Modelos registrados por MLflow en el entrenamiento final (si se incluye en el repo). |

## Entorno de ejecución

- El notebook se ejecutó en **Google Colab con una GPU NVIDIA T4**.
- Todos los tiempos reportados (segundos por época, tiempo hasta converger, tiempo de búsqueda) se midieron con `time.perf_counter()` en esa GPU; en otro hardware los valores absolutos cambiarán, aunque no las comparaciones relativas.
- Funciona también en CPU, pero es **bastante más lento**.
- Referencia de tiempo en la T4: la búsqueda del MLP (201 trials) tomó unos 1.900 s y la del baseline softmax (201 trials), unos 950 s.

## Cómo ejecutarlo

### Opción A: Google Colab

1. Subir `TP2_MLP.ipynb` a Colab (o abrirlo desde Drive/GitHub).
2. Activar la GPU: *Entorno de ejecución → Cambiar tipo de entorno de ejecución → T4 GPU*.
3. Ejecutar todas las celdas en orden. La primera celda instala `optuna` y `mlflow`; el resto de las librerías ya viene en Colab.
4. Al final (sección 15) se genera `artefactos_mlp.zip` y Colab lo descarga automáticamente.

### Opción B: ejecución local

```bash
python -m venv .venv
source .venv/bin/activate        # En Windows: .venv\Scripts\activate

# Instalar PyTorch según el sistema/CUDA (https://pytorch.org/get-started/locally/)
pip install -r requirements.txt

jupyter notebook TP2_MLP.ipynb
```

La primera celda (`!pip install -q optuna mlflow`) puede omitirse si ya se instalaron los requirements.

### Notas

- El notebook descarga automáticamente el dataset (`fetch_20newsgroups`) y los recursos de NLTK (`stopwords`, `wordnet`, `omw-1.4`, `punkt`, `punkt_tab`), por lo que se necesita conexión a internet la primera vez.
- Las semillas están fijadas (`SEED = 42`, `SEEDS = [42, 123, 2024, 7, 99]` para los entrenamientos finales), pero los resultados pueden variar levemente entre ejecuciones por el no determinismo de la GPU.
- Los estudios de Optuna se guardan en `results/optuna_*.db` y **se reanudan** si la sesión se interrumpe. Para repetir la búsqueda desde cero hay que borrar esos archivos.
- El preprocesamiento (lo más lento) se precalcula una sola vez para las 6 combinaciones posibles de stopwords y normalización, y la vectorización TF-IDF se cachea; la primera ejecución de esa celda puede tardar unos minutos.

## Contenido del notebook

| Sección | Qué hace |
|---|---|
| **0. Setup** | Instalación de dependencias, importaciones, semillas, detección de GPU y backend de MLflow. |
| **1. Datos** | Carga de las particiones oficiales train/test de 20 Newsgroups (sin encabezados, firmas ni citas). El test no se usa en la búsqueda. |
| **2. Preprocesamiento** | Tokenización configurable: minúsculas, stopwords, stemming o lematización. |
| **3. Vectorización** | TF-IDF con rango de n-gramas, `max_features` y `min_df` configurables, ajustado solo con train. |
| **4. Modelo** | `MLP` que generaliza la regresión softmax: 0 a 3 capas ocultas, distintas activaciones y dropout. Con 0 capas ocultas equivale exactamente a la regresión softmax. |
| **5. Entrenamiento** | Bucle de entrenamiento con Adam, RMSprop o SGD; métricas por época, tiempos, época de convergencia, early stopping y pruning. |
| **6. Búsqueda con Optuna** | Un estudio por optimizador (Adam, RMSprop, SGD), 67 trials cada uno (201 en total), con `MedianPruner`. Espacio: preprocesamiento, `batch_size`, arquitectura, activación, dropout, weight decay y los hiperparámetros propios de cada optimizador (incluidos sus momentos de 1.er y 2.º orden). |
| **7. Experimento de overfitting** | Grilla de 1, 2 y 3 capas × 64, 256 y 1024 neuronas, sin regularización, para encontrar la configuración **más chica que sobreajusta**. Guarda el resultado en `overfit_config.json`. |
| **8. Entrenamiento final** | Entrena el mejor MLP con todo el train y lo evalúa en el test real. |
| **9. Evaluación final** | Accuracy, `classification_report`, matriz de confusión y confusiones más frecuentes entre clases. |
| **10. Baseline softmax** | Regresión softmax con el mismo protocolo, espacio de búsqueda, presupuesto, semilla y hardware. |
| **11. Análisis de la búsqueda** | Pruning y tiempo ahorrado, mejor trial, efecto del `lr` por optimizador, rendimiento y tiempos por grupo, vectorización, chequeo de bordes del espacio e importancia de hiperparámetros (fANOVA). |
| **12. Efecto de cada hiperparámetro** | Experimentos de un factor por vez desde la mejor configuración (3 semillas): `lr`, `batch_size`, vectorización y optimizador. |
| **13. Multi-semilla y robustez** | Comparación softmax vs MLP con 5 semillas (con test t pareado) y repetición de la búsqueda con otras semillas del sampler. |
| **14. Conclusiones** | Resultados, búsqueda y pruning, efecto de cada hiperparámetro, overfitting, robustez y limitaciones. |
| **15. Exportación** | Guarda las tablas finales y genera `artefactos_mlp.zip` con `results/`, `runs/`, `mlruns/` y `mlflow.db`. |

## Resultados principales

Evaluación en el test oficial, media ± desvío entre 5 semillas:

| | Softmax | MLP |
|---|---|---|
| Accuracy en test | 0.6837 ± 0.0005 | **0.7033 ± 0.0013** |
| Macro-F1 en test | 0.6735 ± 0.0005 | **0.6943 ± 0.0013** |
| Parámetros | 400.020 | 2.562.708 |
| Mejor `val_acc` de la búsqueda | 0.7556 | 0.7649 |

- El MLP supera al softmax por unos 2 puntos de accuracy, de forma consistente en las 5 semillas (t pareado = 30,05).
- El mejor MLP es una red de **una capa de 128 unidades** con `dropout = 0.7`, Adam, `batch_size = 512` y 20.000 características: una arquitectura chica y muy regularizada.
- Adam y RMSprop rinden parecido en validación (0.7649 y 0.7631); SGD queda unos 5 puntos por debajo.
- El pruning cortó el 36 % de los trials del MLP.
- Más detalle (por clase, tiempos, efecto de cada hiperparámetro y limitaciones) en la sección 14 del notebook.

## Relación con el TP3

La sección 7 busca la configuración más chica que sobreajusta el problema y la guarda en `results/overfit_config.json`. Con el criterio usado (`train_acc ≥ 0.97` y `val_loss` final al menos 1,5 veces su mínimo), esa configuración es una red de **3 capas × 64 neuronas** (1.289.684 parámetros), con ReLU, Adam, `lr = 1e-3` y `batch_size = 64`, sin dropout ni weight decay.

Ese archivo es la entrada del **TP3 (regularización)**, que estudia cómo L1, L2, dropout y Batch Normalization cambian el comportamiento de esa red. El `overfit_config.json` del TP3 proviene de una nueva corrida de este notebook, aplicando las recomendaciones y correcciones generales de clase (varias semillas, medición de tiempos, etc.).
