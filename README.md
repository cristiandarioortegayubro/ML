# Machine Learning con PySpark en Databricks

🎓 **Repositorio educativo** con notebooks teóricos y prácticos de Machine Learning usando PySpark ML en Databricks, organizado según los **tres paradigmas fundamentales** de ML.

## 🚀 ¿Nuevo Aquí?

👉 **[LEE PRIMERO: GUÍA DE INICIO RÁPIDO](GUIA_INICIO_RAPIDO.md)** 👈

La guía incluye:
* ⏱️ Plan de estudio paso a paso (15-20 horas)
* ✅ Checklist de progreso
* 🛑 Solución de errores comunes
* 📚 Recursos complementarios
* 🎯 Orden recomendado de notebooks

---

## 📚 Contenido

Este repositorio contiene **21 notebooks completos** divididos en **teoría** y **práctica**, cubriendo desde fundamentos hasta producción con MLOps: **Fundamentos**, **Supervisado**, **No Supervisado**, **Por Refuerzo** y **AutoML/MLOps**.

### 📂 Estructura del Proyecto

```
ML/
├── Fundamentos/
│   ├── 01_Introduccion_Machine_Learning.ipynb  [TEORIA]
│   ├── 02_Matematicas_Esenciales.ipynb         [TEORIA]
│   └── README.md
│
├── Aprendizaje Supervisado/
│   ├── Clasificacion/
│   │   ├── Teoria_Arboles_Decision.ipynb       [TEORIA]
│   │   ├── Teoria_Random_Forest.ipynb          [TEORIA]
│   │   ├── Arbol_Decision_Clasificacion.ipynb  [PRACTICA]
│   │   ├── Random_Forest_Clasificacion.ipynb   [PRACTICA]
│   │   └── GUIA_CLASIFICACION.md
│   │
│   ├── Regresion/
│   │   ├── Teoria_Regresion.ipynb              [TEORIA]
│   │   ├── Regresion_Lineal_Multiple.ipynb     [PRACTICA]
│   │   ├── Arbol_Decision_Regresion.ipynb      [PRACTICA]
│   │   ├── Random_Forest_Regresion.ipynb       [PRACTICA]
│   │   └── README.md
│   │
│   └── README.md
│
├── Aprendizaje No Supervisado/
│   ├── Clustering/
│   │   ├── Teoria_Clustering.ipynb             [TEORIA]
│   │   ├── KMeans_Clustering.ipynb             [PRACTICA]
│   │   └── README.md
│   │
│   └── README.md
│
├── Aprendizaje por Refuerzo/
│   ├── Teoria_Reinforcement_Learning.ipynb     [TEORIA]
│   ├── Q_Learning_GridWorld.ipynb              [PRACTICA]
│   └── README.md
│
└── README.md (este archivo)
```

---

## 🎯 Tres Paradigmas de Machine Learning

### 1️⃣ Aprendizaje Supervisado (`Aprendizaje Supervisado/`)

**Definición**: Aprender de datos **etiquetados** para predecir respuestas correctas.

* **Carpetas**: `Clasificacion/` y `Regresion/`
* **Datos**: Pares (entrada, salida deseada)
* **Objetivo**: Predecir etiqueta o valor numérico
* **Ejemplos**:
  - **Clasificación**: Predicción de churn (Sí/No), detección de fraude
  - **Regresión**: Predicción de precios inmobiliarios, demanda
* **Algoritmos**: Decision Trees, Random Forest, Linear Regression, Gradient Boosting

📊 **Total**: 4 notebooks teóricos + 6 notebooks prácticos

---

### 2️⃣ Aprendizaje No Supervisado (`Aprendizaje No Supervisado/`)

**Definición**: Descubrir patrones ocultos en datos **sin etiquetas**.

* **Carpetas**: `Clustering/`
* **Datos**: Sin respuestas correctas conocidas
* **Objetivo**: Encontrar estructura, grupos, relaciones
* **Ejemplos**:
  - **Clustering**: Segmentación de clientes, agrupación de documentos
  - **Reducción de dimensionalidad**: PCA, t-SNE (próximamente)
  - **Detección de anomalías**: Fraude, fallas (próximamente)
* **Algoritmos**: K-Means, DBSCAN, Hierarchical Clustering

📊 **Total**: 1 notebook teórico + 1 notebook práctico

---

### 3️⃣ Aprendizaje por Refuerzo (`Aprendizaje por Refuerzo/`)

**Definición**: Agente aprende mediante **interacción** con entorno, optimizando recompensas acumuladas.

* **Datos**: Secuencias de (estado, acción, recompensa)
* **Objetivo**: Maximizar recompensa a largo plazo
* **Ejemplos**:
  - Juegos (AlphaGo, Atari)
  - Robótica (navegación, manipulación)
  - Finanzas (trading algorítmico)
  - Vehículos autónomos
* **Algoritmos**: Q-Learning, SARSA, DQN, PPO, SAC

📊 **Total**: 1 notebook teórico + 1 notebook práctico

---

## 📖 Notebooks Teóricos (Resumen)

### 0️⃣ Fundamentos (4 notebooks) ✅ **ARMONIZADOS 2026**

Base matemática y metodológica esencial antes de algoritmos específicos. Todos los notebooks han sido modernizados con formato limpio (sin legacy markup).

**01_Introduccion_Machine_Learning.ipynb** (7 celdas markdown)
- Tipos de aprendizaje (supervisado, no supervisado, refuerzo)
- Proceso completo de ML (pipeline)
- Métricas de evaluación (clasificación y regresión)
- Overfitting/Underfitting
- Formato: Markdown moderno con emojis y tablas

**02_Matematicas_Esenciales.ipynb** (8 celdas markdown)
- Álgebra lineal (vectores, matrices, aplicaciones en ML)
- Cálculo (derivadas, gradientes, backpropagation)
- Probabilidad y estadística (Bayes, MLE)
- Optimización (GD, SGD, Mini-Batch, Adam)
- Teoría de la Información (entropía, cross-entropy, KL)
- Formato: Markdown con ecuaciones LaTeX

**03_Preprocesamiento_y_Feature_Engineering.ipynb** (12 celdas: 10 markdown + 2 código)
- Limpieza de datos (valores faltantes, outliers con IQR/Z-Score)
- Encoding (Label, One-Hot, Target)
- Scaling (StandardScaler, MinMaxScaler, RobustScaler)
- Feature Engineering (interacciones, features de fecha, binning)
- Selección de features (Filter, Wrapper, Embedded methods)
- Ejemplos ejecutables con sklearn y pandas
- Formato: Mix moderno de markdown y código Python

**04_Evaluacion_y_Validacion.ipynb** (8 celdas: 5 markdown + 3 código)
- Métricas clasificación (Matriz de confusión, F1, ROC-AUC)
- Validación cruzada (K-Fold, Stratified K-Fold)
- Overfitting/Underfitting: síntomas, diagnóstico con ejemplos
- Bias-Variance Tradeoff con visualizaciones
- Ejemplos ejecutables: fraude detection, regresión
- Formato: Mix moderno de markdown y código Python

### 1️⃣ Aprendizaje Supervisado

#### Clasificación

**Teoria_Arboles_Decision.ipynb** (7 celdas, 311 líneas)
- Algoritmo ID3, entropía, Gini
- Poda, overfitting

**Teoria_Random_Forest.ipynb** (8 celdas, ~400 líneas)
- Ensemble learning, bagging
- Feature importance, OOB error

#### Regresión

**Teoria_Regresion.ipynb** (7 celdas, 336 líneas)
- Regresión lineal (simple, múltiple)
- Regularización (Ridge, Lasso)
- Métodos no lineales

### 2️⃣ Aprendizaje No Supervisado

**Teoria_Clustering.ipynb** (7 celdas, 370 líneas)
- K-Means, DBSCAN, Hierarchical
- Métricas (Silhouette, Elbow)
- Determinación de K

### 3️⃣ Aprendizaje por Refuerzo

**Teoria_Reinforcement_Learning.ipynb** (3 celdas, ~2000 líneas)
- MDP, funciones de valor, Bellman
- Q-Learning, SARSA, DQN
- Exploración vs Explotación
- Aplicaciones modernas

---

## 💻 Notebooks Prácticos (Resumen)

### Clasificación
1. **Arbol_Decision_Clasificacion**: Churn prediction (Decision Tree)
2. **Random_Forest_Clasificacion**: Churn prediction (Random Forest, mejora vs DT)

### Regresión
1. **Regresion_Lineal_Multiple**: Precios inmobiliarios (Linear)
2. **Arbol_Decision_Regresion**: Precios inmobiliarios (Tree)
3. **Random_Forest_Regresion**: Precios inmobiliarios (Random Forest, mejor R²)

### Clustering
1. **KMeans_Clustering**: Segmentación de clientes

### Reinforcement Learning
1. **Q_Learning_GridWorld**: Navegación en cuadrícula (Q-Learning)

---

## 🚀 Orden de Estudio Recomendado

### Para Principiantes (Path Completo):

```
1. Fundamentos/
   ├── 01_Introduccion_Machine_Learning      [1-2 horas]
   └── 02_Matematicas_Esenciales             [2-3 horas]

2. Aprendizaje Supervisado/
   ├── Clasificacion/
   │   ├── Teoria_Arboles_Decision           [1 hora]
   │   ├── Arbol_Decision_Clasificacion      [30 min]
   │   ├── Teoria_Random_Forest              [1 hora]
   │   └── Random_Forest_Clasificacion       [30 min]
   │
   └── Regresion/
       ├── Teoria_Regresion                  [1 hora]
       ├── Regresion_Lineal_Multiple         [30 min]
       ├── Arbol_Decision_Regresion          [30 min]
       └── Random_Forest_Regresion           [30 min]

3. Aprendizaje No Supervisado/
   └── Clustering/
       ├── Teoria_Clustering                 [1 hora]
       └── KMeans_Clustering                 [30 min]

4. Aprendizaje por Refuerzo/
   ├── Teoria_Reinforcement_Learning         [2-3 horas]
   └── Q_Learning_GridWorld                  [1 hora]

TOTAL: ~15-18 horas de estudio
```

### Para Avanzados (Fast Track):

1. Revisar fundamentos rápidamente
2. Saltar directamente a Random Forest (clasificación y regresión)
3. Explorar Reinforcement Learning (Q-Learning, DQN)
4. Experimentar con hiperparámetros y datasets propios

---

## 🎓 Características Educativas

### ✅ Contenido Teórico Riguroso

* **Fórmulas matemáticas**: Todas las ecuaciones clave explicadas
* **Pseudocódigo**: Algoritmos paso a paso
* **Comparaciones**: Ventajas y desventajas de cada método
* **Intuición**: Explicaciones conceptuales claras

### ✅ Código Práctico Production-Ready

* **PySpark ML nativo**: Escalable para Big Data
* **Datasets sintéticos realistas**: 10,000 muestras con reglas de negocio
* **Visualizaciones**: Matplotlib, Seaborn
* **Métricas completas**: Accuracy, RMSE, Silhouette, etc.
* **Comparaciones**: Múltiples modelos lado a lado

### ✅ README Comprensivos

Cada carpeta incluye README con:
* Descripción conceptual
* Lista de notebooks con contenido detallado
* Orden de estudio recomendado
* Aplicaciones reales del mundo empresarial
* Enlaces a recursos externos

---


### 5️⃣ **AutoML y MLOps** ⚡ NUEVO
**Automatización y Producción de Machine Learning**

Aprende a usar **Databricks AutoML** para generar modelos automáticamente e implementar **MLOps** para llevar modelos a producción de forma escalable.

| Notebook | Tipo | Descripción | Celdas |
|----------|------|-------------|--------|
| [Teoria_AutoML.ipynb](AutoML/Teoria_AutoML.ipynb) | Teoría | Conceptos de AutoML, MLOps, herramientas, comparación Manual vs AutoML | 4 |
| [Databricks_AutoML_Clasificacion.ipynb](AutoML/Databricks_AutoML_Clasificacion.ipynb) | Práctica | AutoML para clasificación (churn), comparación con modelo manual, MLflow | 13 |
| [Databricks_AutoML_Regresion.ipynb](AutoML/Databricks_AutoML_Regresion.ipynb) | Práctica | AutoML para regresión, métricas RMSE/MAE/R² | 8 |
| [Genie_Assisted_ML_Pipeline.ipynb](AutoML/Genie_Assisted_ML_Pipeline.ipynb) | Práctica | Pipelines completos con Genie Code como asistente | 8 |
| [MLflow_Experiment_Tracking.ipynb](AutoML/MLflow_Experiment_Tracking.ipynb) | Práctica | Tracking, Model Registry, deployment con MLflow | 7 |

**Conceptos clave:**
* 🤖 **AutoML**: Preprocesamiento automático, feature engineering, optimización de hiperparámetros
* 🏭 **MLOps**: Experiment tracking, Model Registry, Feature Store, Model Serving, Monitoring
* 📊 **Comparación**: Manual vs AutoML - cuándo usar cada uno
* 🔧 **Databricks Stack**: AutoML, MLflow, Feature Store, Model Serving
* 🎯 **Producción**: Despliegue de modelos, monitoring, reentrenamiento automático

**Tiempo estimado:** 5-7 horas (con notebooks completos)

📖 **[Ver README completo](AutoML/README.md)**

## 📊 Estadísticas del Repositorio

| Categoría | Cantidad | Estado |
|-----------|----------|--------|
| **Notebooks teóricos** | 15 | 4 Fundamentos + 4 Supervisado + 1 No Supervisado + 1 Refuerzo + 1 AutoML |
| **Notebooks prácticos** | 11 | 6 Supervisado + 1 No Supervisado + 1 Refuerzo + 4 AutoML/MLOps |
| **Total notebooks** | 26 | **Todos con formato moderno (2026)** |
| **README files** | 10 | Principal + Fundamentos + Supervisado + AutoML + Otros |
| **Guías** | 2 | README.md + GUIA_INICIO_RAPIDO.md |

### Distribución por Paradigma

| Paradigma | Teoría | Práctica | Total |
|-----------|--------|----------|-------|
| **Fundamentos** | 4 | 0 | 4 |
| **Supervisado** | 4 | 6 | 10 |
| **No Supervisado** | 1 | 1 | 2 |
| **Por Refuerzo** | 1 | 1 | 2 |
| **AutoML/MLOps** | 1 | 4 | 5 |

---

## 🛠️ Tecnologías y Herramientas

* **Databricks**: Plataforma unificada de datos e IA
* **Apache Spark**: Procesamiento distribuido a gran escala
* **PySpark ML**: Biblioteca de Machine Learning de Spark
* **Python**: NumPy, Pandas, Matplotlib, Seaborn
* **Serverless Compute**: Ejecución sin gestión de clusters
* **MLflow**: Tracking de experimentos (próximamente)

---

## 💼 Aplicaciones del Mundo Real

### Clasificación
* 📞 Telecom: Predicción de churn
* 🏦 Banking: Detección de fraude
* 📧 Marketing: Spam detection
* 🏥 Healthcare: Diagnóstico de enfermedades

### Regresión
* 🏡 Real Estate: Valoración de propiedades
* 📈 Finance: Predicción de precios de acciones
* 🛒 E-commerce: Forecasting de demanda

### Clustering
* 🛍️ Retail: Segmentación de clientes (RFM)
* 📰 Media: Agrupación de noticias por tema
* 🧬 Bioinformática: Análisis de genes

### Reinforcement Learning
* 🎮 Gaming: AlphaGo, OpenAI Five
* 🤖 Robótica: Navegación, manipulación
* 💰 Trading: Optimización de estrategias
* 🚗 Vehículos autónomos

---

## 📚 Recursos Adicionales

### Documentación oficial
* [PySpark ML Guide](https://spark.apache.org/docs/latest/ml-guide.html)
* [Databricks ML Documentation](https://docs.databricks.com/machine-learning/index.html)

### Cursos recomendados
* **Coursera**: Machine Learning (Andrew Ng)
* **Fast.ai**: Practical Deep Learning
* **UC Berkeley CS 285**: Deep Reinforcement Learning

### Libros
* **"Hands-On Machine Learning"** - Aurélien Géron
* **"Pattern Recognition and Machine Learning"** - Christopher Bishop
* **"Reinforcement Learning: An Introduction"** - Sutton & Barto

---

## 🤝 Contribuciones

Este es un repositorio educativo en constante evolución. Áreas de expansión futura:

### Próximos temas:
* ✅ **Supervisado**: Gradient Boosting (XGBoost, LightGBM), SVM, Logistic Regression
* ✅ **No Supervisado**: PCA, t-SNE, DBSCAN, Detección de anomalías
* ✅ **Refuerzo**: SARSA, DQN, Policy Gradients, Actor-Critic
* ✅ **Deep Learning**: Redes neuronales, CNNs, RNNs, Transformers
* ✅ **MLOps**: MLflow, Feature Store, Model Serving

---

## 📄 Licencia

Este repositorio es de uso educativo. Se permite copiar, modificar y distribuir con atribución.

---

## 👨‍💻 Autor

Creado con 💙 para la comunidad de Data Science y Machine Learning.

**¡Explora, aprende y experimenta!** 🚀✨
