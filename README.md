# Machine Learning con PySpark en Databricks

🎓 **Repositorio educativo** con ejemplos prácticos de Machine Learning usando PySpark ML en Databricks.

## 📚 Contenido

Este repositorio contiene **4 notebooks completos** que cubren los tres tipos principales de Machine Learning:

### 📂 Estructura del Proyecto

```
ML/
├── Clasificacion/
│   ├── Arbol_Decision_Clasificacion.py
│   └── README.md
├── Regresion/
│   ├── Arbol_Decision_Regresion.py
│   ├── Regresion_Lineal_Multiple.py
│   └── README.md
├── Clustering/
│   ├── KMeans_Clustering.py
│   └── README.md
└── README.md (este archivo)
```

### 1️⃣ Clasificación - Predicción de Churn

**Notebook**: `Clasificacion/Arbol_Decision_Clasificacion.py`

* **Algoritmo**: Decision Tree Classifier
* **Problema**: Predecir qué clientes abandonarán una empresa de telecomunicaciones
* **Dataset**: 10,000 clientes (sintético)
* **Métricas**: Accuracy, Precision, Recall, F1-Score, AUC-ROC

### 2️⃣ Regresión - Predicción de Precios Inmobiliarios

**Notebook 1**: `Regresion/Arbol_Decision_Regresion.py` (Decision Tree Regressor)
**Notebook 2**: `Regresion/Regresion_Lineal_Multiple.py` (Linear Regression)

* **Problema**: Predecir precio de venta de propiedades
* **Dataset**: 10,000 propiedades (sintético)
* **Métricas**: RMSE, MAE, R²

### 3️⃣ Clustering - Segmentación de Clientes

**Notebook**: `Clustering/KMeans_Clustering.py`

* **Algoritmo**: K-Means Clustering
* **Problema**: Segmentar clientes de e-commerce
* **Dataset**: 5,000 clientes (sintético)
* **Métricas**: WSSSE, Silhouette Score

---

## 🚀 Guía de Inicio Rápido

### Paso 1: Crear cuenta en Databricks Community Edition (GRATIS)

1. Visita: **https://community.cloud.databricks.com/login.html**
2. Haz clic en **"Sign up for Community Edition"**
3. Completa el formulario y verifica tu email
4. Inicia sesión

### Paso 2: Crear un Cluster

1. Menú lateral → **"Compute"**
2. **"Create Cluster"**
3. Configuración:
   - **Cluster Name**: "ML-Cluster"
   - **Databricks Runtime**: Selecciona versión con ML (ej: "Runtime 14.3 LTS ML")
4. **"Create Cluster"** y espera 3-5 minutos

### Paso 3: Clonar este repositorio

**Opción A: Usando Git (Recomendado)**

1. Menú **"Workspace"**
2. Clic derecho en tu carpeta de usuario
3. **"Create" → "Repo"**
4. URL: `https://github.com/cristiandarioortegayubro/ML`
5. **"Create Repo"**

**Opción B: Descarga manual**

1. Descarga este repo como ZIP desde GitHub
2. Descomprime
3. En Databricks: Clic derecho → **"Import"**
4. Arrastra los archivos `.py`

### Paso 4: Ejecutar notebooks

1. Abre cualquier notebook (ej: `ML/Clasificacion/Arbol_Decision_Clasificacion`)
2. Asegúrate de que tu cluster esté seleccionado y activo
3. Ejecuta las celdas:
   - **Shift + Enter**: Celda por celda
   - **Menú "Run" → "Run All"**: Todas a la vez
4. Observa resultados, gráficos y métricas

---

## 💻 Requisitos

### Databricks Community Edition (Gratis)

* **Recursos**: 15 GB RAM, 2 cores
* **Limitaciones**: Cluster se detiene tras 2h inactividad
* **Suficiente para**: Todos estos notebooks

---

## 📚 Conceptos Cubiertos

* **Aprendizaje Supervisado**: Clasificación y Regresión
* **Aprendizaje No Supervisado**: Clustering
* **Preparación de datos**: StringIndexer, VectorAssembler, StandardScaler
* **Evaluación**: Métricas de clasificación, regresión y clustering
* **Visualización**: Matplotlib, Seaborn

---

## 💬 Preguntas Frecuentes

**¿Necesito experiencia previa en Spark?**
No. Los notebooks incluyen explicaciones detalladas.

**¿Puedo usar mis propios datos?**
Sí. Sube tu CSV y adapta la preparación de datos.

**¿Por qué archivos .py en lugar de .ipynb?**
Databricks usa formato Source (.py) pero se editan como notebooks interactivos. Puedes exportar a .ipynb desde el menú.

---

## 📚 Recursos

* [PySpark ML Guide](https://spark.apache.org/docs/latest/ml-guide.html)
* [Databricks Documentation](https://docs.databricks.com/)
* [Databricks Academy (Gratis)](https://www.databricks.com/learn/training)

---

## ✅ Checklist

- [ ] Creé cuenta en Databricks
- [ ] Cloné el repositorio
- [ ] Inicié mi cluster
- [ ] Ejecuté notebook de Clasificación
- [ ] Ejecuté notebooks de Regresión
- [ ] Ejecuté notebook de Clustering
- [ ] Experimenté con mis propios datos

---

**¡Mucho éxito en tu viaje de Machine Learning!** 🎓🚀
