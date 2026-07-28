# Machine Learning con PySpark en Databricks

🎓 **Repositorio educativo** con notebooks teóricos y prácticos de Machine Learning usando PySpark ML en Databricks.

## 📚 Contenido

Este repositorio contiene **9 notebooks completos** divididos en **teoría** y **práctica**, cubriendo fundamentos y los tres tipos principales de Machine Learning.

### 📂 Estructura del Proyecto

```
ML/
├── Fundamentos/
│   ├── 01_Introduccion_Machine_Learning.ipynb  [TEORIA]
│   ├── 02_Matematicas_Esenciales.ipynb         [TEORIA]
│   └── README.md
├── Clasificacion/
│   ├── Teoria_Arboles_Decision.ipynb           [TEORIA]
│   ├── Arbol_Decision_Clasificacion.ipynb      [PRACTICA]
│   └── README.md
├── Regresion/
│   ├── Teoria_Regresion.ipynb                  [TEORIA]
│   ├── Arbol_Decision_Regresion.ipynb          [PRACTICA]
│   ├── Regresion_Lineal_Multiple.ipynb         [PRACTICA]
│   └── README.md
├── Clustering/
│   ├── Teoria_Clustering.ipynb                 [TEORIA]
│   ├── KMeans_Clustering.ipynb                 [PRACTICA]
│   └── README.md
└── README.md (este archivo)
```

### 📖 Notebooks Teóricos (1,935 líneas de contenido académico)

#### 0️⃣ Fundamentos

**01_Introduccion_Machine_Learning.ipynb** (7 celdas, 425 líneas)
- Tipos de aprendizaje (supervisado, no supervisado, refuerzo)
- Proceso completo de ML (pipeline)
- Métricas de evaluación (clasificación y regresión)
- Overfitting/Underfitting y regularización
- Ingeniería de características

**02_Matematicas_Esenciales.ipynb** (8 celdas, 493 líneas)
- Álgebra lineal (vectores, matrices, operaciones)
- Cálculo (derivadas, gradientes, backpropagation)
- Probabilidad y estadística (Bayes, distribuciones, MLE)
- Optimización (GD, SGD, Adam)
- Teoría de la información (entropía, cross-entropy)

#### 1️⃣ Clasificación - Predicción de Churn

**Teoría**: `Teoria_Arboles_Decision.ipynb` (7 celdas, 311 líneas)
- Algoritmo ID3, entropía, índice Gini
- Poda pre/post, control de overfitting
- Ventajas y desventajas

**Práctica**: `Arbol_Decision_Clasificacion.ipynb`
- **Algoritmo**: Decision Tree Classifier
- **Problema**: Predicción de churn (abandono de clientes)
- **Dataset**: 10,000 clientes de telecomunicaciones (sintético)
- **Métricas**: Accuracy, Precision, Recall, F1-Score, AUC-ROC

#### 2️⃣ Regresión - Predicción de Precios Inmobiliarios

**Teoría**: `Teoria_Regresion.ipynb` (7 celdas, 336 líneas)
- Regresión lineal simple y múltiple
- Regularización (Ridge, Lasso, Elastic Net)
- Métodos no lineales

**Práctica**: 2 notebooks
- `Arbol_Decision_Regresion.ipynb` (Decision Tree Regressor)
- `Regresion_Lineal_Multiple.ipynb` (Linear Regression)
- **Problema**: Predicción de precios de propiedades
- **Dataset**: 10,000 propiedades (sintético)
- **Métricas**: RMSE, MAE, R²

#### 3️⃣ Clustering - Segmentación de Clientes

**Teoría**: `Teoria_Clustering.ipynb` (7 celdas, 370 líneas)
- K-Means, métricas de similitud
- Selección de k (método del codo, silueta)
- DBSCAN, GMM, hierarchical clustering

**Práctica**: `KMeans_Clustering.ipynb`
- **Algoritmo**: K-Means Clustering
- **Problema**: Segmentación de clientes de e-commerce
- **Dataset**: 5,000 clientes (sintético)
- **Métricas**: WSSSE, Silhouette Score

---

## 🚀 Guía de Inicio Rápido

### Paso 1: Crear cuenta en Databricks Free Edition (GRATIS)

1. Visita: **https://community.cloud.databricks.com/login.html**
2. Haz clic en **"Sign up for Free Edition"**
3. Completa el formulario y verifica tu email
4. Inicia sesión

### Paso 2: Clonar este repositorio

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

### Databricks Free Edition (Gratis)

* **Recursos**: 15 GB RAM, 2 cores
* **Limitaciones**: Cluster se detiene tras 2h inactividad
* **Suficiente para**: Todos estos notebooks

---

## 📚 Conceptos Cubiertos

* **Fundamentos teóricos**: Tipos de aprendizaje, matemáticas esenciales, métricas
* **Aprendizaje Supervisado**: Clasificación y Regresión (teoría y práctica)
* **Aprendizaje No Supervisado**: Clustering (teoría y práctica)
* **Preparación de datos**: StringIndexer, VectorAssembler, StandardScaler
* **Evaluación**: Métricas de clasificación, regresión y clustering
* **Visualización**: Matplotlib, Seaborn

---

## 💬 Preguntas Frecuentes

**¿Necesito experiencia previa en Spark?**
No. Los notebooks incluyen explicaciones detalladas.

**¿Puedo usar mis propios datos?**
Sí. Sube tu CSV y adapta la preparación de datos.

**¿Por qué notebooks .ipynb?**
Todos los notebooks están en formato Jupyter (.ipynb) para compatibilidad total con GitHub y Databricks.

---

## 📚 Recursos

* [PySpark ML Guide](https://spark.apache.org/docs/latest/ml-guide.html)
* [Databricks Documentation](https://docs.databricks.com/)
* [Databricks Academy (Gratis)](https://www.databricks.com/learn/training)

---

## ✅ Checklist

- [ ] Creé cuenta en Databricks
- [ ] Cloné el repositorio
- [ ] Ejecuté notebook de Clasificación
- [ ] Ejecuté notebooks de Regresión
- [ ] Ejecuté notebook de Clustering
- [ ] Experimenté con mis propios datos

---

**¡Mucho éxito en tu viaje de Machine Learning!** 🎓🚀
