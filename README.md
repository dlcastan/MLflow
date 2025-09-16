
# 📨 Clasificador de SPAM (TF‑IDF + Naive Bayes) con MLflow

Este proyecto entrena y prueba un **clasificador de SPAM** usando **TF‑IDF** + **Multinomial Naive Bayes**, registrando **métricas, artefactos y el modelo** en **MLflow**. Incluye un cuaderno de entrenamiento y un cuaderno para pruebas rápidas del modelo.

---

## 📦 Contenidos del repo (archivos provistos)

- `spam_dataset.csv` → dataset con **1000 filas** y **2 columnas**:
  - `message_content` *(texto)*: contenido del mensaje.
  - `is_spam` *(0/1)*: etiqueta binaria (1 = SPAM, 0 = no SPAM).
- `spam_classifier_mlflow.ipynb` → notebook de **entrenamiento + tracking en MLflow**.
- `test_mlflow_model.ipynb` → notebook para **cargar y probar** un modelo registrado en MLflow (generado en esta sesión).

> Si cambiás los nombres de columna, mantené consistencia entre entrenamiento y predicción (por ejemplo, si el pipeline espera `message_content`, usalo igual en las pruebas y en la API).

---

## 🛠 Requisitos

- Python 3.9+ (recomendado)
- Librerías: `mlflow`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`
- (Opcional) `conda` o `venv` para aislar el entorno

Instalación rápida:
```bash
pip install mlflow scikit-learn pandas numpy matplotlib
```

---

## 🚀 Flujo de trabajo

1) **Entrenamiento** (en `spam_classifier_mlflow.ipynb`)
   - Carga y limpieza del dataset `spam_dataset.csv`.
   - Vectorización con **TF‑IDF**.
   - Entrenamiento con **Multinomial Naive Bayes**.
   - **Validación** (F1 con CV).
   - **Evaluación** en test (accuracy, precision, recall, f1).
   - **MLflow**: log de métricas, artefactos y modelo (incluye el pipeline con TF‑IDF + NB).

2) **Pruebas del modelo** (en `test_mlflow_model.ipynb`)
   - Recupera un `run_id` o toma el **último run** de un experimento.
   - Carga el modelo vía `mlflow.pyfunc.load_model(...)`.
   - Hace **predicciones de ejemplo** con la columna de entrada `message_content` (ajustable).
   - Si existe `test.csv`, calcula **classification_report** y **confusion_matrix**.

3) **(Opcional) Servir como API** con MLflow Model Server.

---

## ▶️ Cómo entrenar y registrar en MLflow

Abrí `spam_classifier_mlflow.ipynb` y ejecutá las celdas. Asegurate de que el logging de MLflow apunte a donde querés (por defecto, `mlruns` local).  
En el notebook se loguea el modelo bajo un subpath (típicamente `model`).

**Estructura de artefactos** (típica):
```
mlruns/
 └─ <experiment_id>/
     └─ <run_id>/
         ├─ artifacts/
         │   └─ model/         # modelo pyfunc (incluye conda/requirements)
         └─ params/metrics/...
```

---

## 🧪 Cómo probar el modelo (notebook listo)

Usá `test_mlflow_model.ipynb`. Podés setear variables de entorno para personalizar:

- `MLFLOW_RUN_ID` → si querés cargar un run específico.
- `MLFLOW_EXPERIMENT_NAME` → por defecto `"Default"`.
- `MLFLOW_MODEL_SUBPATH` → por defecto `"model"`.
- `INPUT_COLUMN_NAME` → por defecto `"text"`, **cambialo a** `"message_content"` para este dataset.
- `TEST_CSV_PATH` → ruta a `test.csv` (si querés evaluar).

Ejemplo al ejecutar localmente (Linux/macOS):
```bash
export INPUT_COLUMN_NAME=message_content
jupyter lab  # o jupyter notebook
```

En Windows (PowerShell):
```powershell
$env:INPUT_COLUMN_NAME="message_content"
jupyter lab
```

---

## 🌐 Servir el modelo como API local

1) Identificá tu `MODEL_URI` (el notebook de pruebas lo imprime, suele ser `runs:/<RUN_ID>/model`).  
2) Levantá el servidor:
```bash
mlflow models serve -m "runs:/<RUN_ID>/model" -p 5000 --enable-mlserver
```
3) Probá con `curl` (**nota**: ajustá el nombre de la columna a `message_content`):
```bash
curl -X POST http://127.0.0.1:5000/invocations   -H "Content-Type: application/json"   -d '{
        "dataframe_split": {
          "columns": ["message_content"],
          "data": [
            ["¡Ganaste un premio! Transferí tus datos bancarios para recibirlo"],
            ["Hola, ¿confirmamos la reunión a las 11?"]
          ]
        }
      }'
```

La respuesta será algo como:
```json
{"predictions":[1,0]}
```
(donde 1 = SPAM, 0 = no SPAM, si usaste esa codificación).

---

## 📚 Sugerencias de evaluación

- **Métricas**: `f1`, `precision`, `recall`, `accuracy`. Para datasets desbalanceados, priorizá `f1`/`recall`.
- **Matriz de confusión**: para visualizar falsos positivos/negativos.
- **Curva Precision‑Recall**: útil si el umbral de decisión es relevante para tu caso de negocio.

---

## 🧩 Estructura mínima sugerida

```
.
├─ data/
│  └─ spam_dataset.csv
├─ notebooks/
│  ├─ spam_classifier_mlflow.ipynb
│  └─ test_mlflow_model.ipynb
├─ mlruns/              # (se crea al entrenar)
└─ README.md
```

---

## ❗️Problemas comunes

- **Versiones de librerías**: si al cargar falla, replicá el entorno del modelo (ver `conda.yaml`/`requirements.txt` en `artifacts/model`).  
- **Nombre de columna**: el pipeline espera una columna específica (aquí `message_content`). Si mandás `text`, fallará o predecirá mal.  
- **Vectorizador**: asegurate de haber logueado el **pipeline completo** (TF‑IDF + clasificador). Si logueás solo el clasificador, `predict` fallará.  
- **Dataset desbalanceado**: considerá `class_weight`, `threshold tuning` o **resampling**.

---

## 🔒 (Opcional) Registrar en Model Registry

Si usás el Registry, podés cargar por nombre y etapa:
```python
import mlflow
model = mlflow.pyfunc.load_model("models:/spam-classifier/Production")
```

---

## 📄 Licencia

MIT (o la que elijas).

---

## 👋 Contacto / mantenimiento

- Autor: Diego (ajustar si corresponde).
- PRs y mejoras bienvenidas.
