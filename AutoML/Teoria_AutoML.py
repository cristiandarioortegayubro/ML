# Databricks notebook source
# DBTITLE 1,# AutoML y MLOps - Teoría
# MAGIC %md
# MAGIC # AutoML y MLOps - Automatización de Machine Learning
# MAGIC
# MAGIC ## 🎯 Definición
# MAGIC
# MAGIC **Automated Machine Learning (AutoML)** es el proceso de **automatizar las tareas** del ciclo de vida de Machine Learning, desde la preparación de datos hasta el despliegue del modelo, permitiendo que usuarios con menos experiencia técnica puedan crear modelos de alta calidad.
# MAGIC
# MAGIC **MLOps (Machine Learning Operations)** es el conjunto de prácticas para desplegar, monitorear y mantener modelos de ML en **producción** de forma confiable y escalable.
# MAGIC
# MAGIC ### La Evolución del ML
# MAGIC
# MAGIC ```
# MAGIC TRADICIONAL (Manual)              AUTOML (Automatizado)           MLOPS (Producción)
# MAGIC      ↓                                   ↓                              ↓
# MAGIC 👨‍💻 Científico de datos          🤖 Automatización               🏭 Operaciones
# MAGIC Codifica cada paso              Sistema decide                  Deploy continuo
# MAGIC Semanas de trabajo              Horas/Minutos                   Monitoreo 24/7
# MAGIC Modelos ad-hoc                  Mejores prácticas               Pipelines robustos
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🔄 El Ciclo Completo de ML
# MAGIC
# MAGIC ### Enfoque Manual (Lo que hemos hecho hasta ahora)
# MAGIC
# MAGIC ```
# MAGIC 1. 📊 Exploración de datos (EDA)
# MAGIC    ├─ Estadísticas descriptivas
# MAGIC    ├─ Visualizaciones
# MAGIC    └─ Detección de outliers
# MAGIC    
# MAGIC 2. 🧹 Preprocesamiento
# MAGIC    ├─ Limpieza de datos
# MAGIC    ├─ Imputación de valores faltantes
# MAGIC    ├─ Encoding de categóricas
# MAGIC    └─ Normalización/Estandarización
# MAGIC    
# MAGIC 3. 🔧 Feature Engineering
# MAGIC    ├─ Creación de features
# MAGIC    ├─ Selección de features
# MAGIC    └─ Transformaciones
# MAGIC    
# MAGIC 4. 🎯 Selección de modelo
# MAGIC    ├─ Probar múltiples algoritmos
# MAGIC    └─ Comparar resultados
# MAGIC    
# MAGIC 5. ⚙️ Optimización de hiperparámetros
# MAGIC    ├─ Grid Search
# MAGIC    ├─ Random Search
# MAGIC    └─ Bayesian Optimization
# MAGIC    
# MAGIC 6. ✅ Evaluación
# MAGIC    ├─ Métricas
# MAGIC    ├─ Validación cruzada
# MAGIC    └─ Análisis de errores
# MAGIC    
# MAGIC 7. 🚀 Despliegue
# MAGIC    └─ Producción
# MAGIC
# MAGIC TIEMPO: Días/Semanas
# MAGIC ```
# MAGIC
# MAGIC ### Enfoque AutoML
# MAGIC
# MAGIC ```
# MAGIC 📥 Input: Dataset + Target
# MAGIC     ↓
# MAGIC 🤖 AutoML Engine:
# MAGIC     ├─ Preprocesamiento automático
# MAGIC     ├─ Feature engineering automático
# MAGIC     ├─ Prueba de múltiples algoritmos
# MAGIC     ├─ Hyperparameter tuning
# MAGIC     └─ Ensemble de mejores modelos
# MAGIC     ↓
# MAGIC 📤 Output: Mejor modelo + Notebook explicativo
# MAGIC
# MAGIC TIEMPO: Minutos/Horas
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🤖 ¿Qué Automatiza AutoML?
# MAGIC
# MAGIC ### 1️⃣ **Preprocesamiento de Datos**
# MAGIC
# MAGIC * **Detección automática de tipos**: Numérico, categórico, texto, fecha
# MAGIC * **Imputación inteligente**: 
# MAGIC   - Numéricos: Media, mediana, KNN imputation
# MAGIC   - Categóricos: Moda, nueva categoría "Missing"
# MAGIC * **Encoding automático**:
# MAGIC   - One-Hot Encoding para categóricas de baja cardinalidad
# MAGIC   - Target Encoding para alta cardinalidad
# MAGIC   - Embeddings para texto
# MAGIC * **Scaling automático**: StandardScaler, MinMaxScaler según algoritmo
# MAGIC * **Manejo de outliers**: Detección y tratamiento
# MAGIC
# MAGIC ### 2️⃣ **Feature Engineering**
# MAGIC
# MAGIC * **Creación de features**:
# MAGIC   - Features de fecha/tiempo (día semana, mes, trimestre)
# MAGIC   - Interacciones entre variables
# MAGIC   - Agregaciones estadísticas
# MAGIC * **Selección de features**:
# MAGIC   - Eliminación de correlaciones altas
# MAGIC   - Importancia de features
# MAGIC   - Recursive Feature Elimination (RFE)
# MAGIC * **Transformaciones**:
# MAGIC   - Log, sqrt, polynomial features
# MAGIC   - Binning de variables continuas
# MAGIC
# MAGIC ### 3️⃣ **Selección de Modelos**
# MAGIC
# MAGIC AutoML prueba automáticamente múltiples algoritmos:
# MAGIC
# MAGIC **Para Clasificación:**
# MAGIC * Logistic Regression
# MAGIC * Decision Trees
# MAGIC * Random Forest
# MAGIC * Gradient Boosted Trees (XGBoost, LightGBM)
# MAGIC * Neural Networks
# MAGIC * Support Vector Machines
# MAGIC
# MAGIC **Para Regresión:**
# MAGIC * Linear Regression (Ridge, Lasso, ElasticNet)
# MAGIC * Decision Tree Regressor
# MAGIC * Random Forest Regressor
# MAGIC * XGBoost, LightGBM Regressor
# MAGIC * Neural Networks
# MAGIC
# MAGIC ### 4️⃣ **Optimización de Hiperparámetros**
# MAGIC
# MAGIC **Métodos de búsqueda:**
# MAGIC
# MAGIC #### Grid Search
# MAGIC ```python
# MAGIC # Probar todas las combinaciones
# MAGIC params = {
# MAGIC     'n_estimators': [100, 200, 300],
# MAGIC     'max_depth': [5, 10, 15],
# MAGIC     'learning_rate': [0.01, 0.1, 0.3]
# MAGIC }
# MAGIC # Total: 3 × 3 × 3 = 27 combinaciones
# MAGIC ```
# MAGIC
# MAGIC #### Random Search
# MAGIC ```python
# MAGIC # Probar N combinaciones aleatorias
# MAGIC # Más eficiente que Grid Search
# MAGIC # Explora mejor el espacio de hiperparámetros
# MAGIC ```
# MAGIC
# MAGIC #### Bayesian Optimization ⭐
# MAGIC ```python
# MAGIC # El más sofisticado
# MAGIC # Usa resultados previos para decidir qué probar
# MAGIC # Converge más rápido al óptimo
# MAGIC # Usado por Databricks AutoML
# MAGIC ```
# MAGIC
# MAGIC **Proceso:**
# MAGIC
# MAGIC $$\text{Objetivo: Maximizar} \quad f(\theta) = \text{Validation Metric}(\text{Model}_{\theta})$$
# MAGIC
# MAGIC Donde $\theta$ son los hiperparámetros.
# MAGIC
# MAGIC ### 5️⃣ **Ensemble Methods**
# MAGIC
# MAGIC AutoML a menudo crea **ensembles** de los mejores modelos:
# MAGIC
# MAGIC * **Voting Classifier/Regressor**: Promedio de predicciones
# MAGIC * **Stacking**: Meta-modelo que aprende de otros modelos
# MAGIC * **Weighted Ensemble**: Pesos óptimos para cada modelo
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🏭 MLOps: De Experimento a Producción
# MAGIC
# MAGIC ### El Problema del "Research-Production Gap"
# MAGIC
# MAGIC ```
# MAGIC ❌ PROBLEMA COMÚN:
# MAGIC
# MAGIC Científico de Datos        →    Ingeniero de Software
# MAGIC "Funciona en mi notebook"       "No puedo desplegarlo"
# MAGIC
# MAGIC 📓 Jupyter Notebook              🏭 Producción
# MAGIC ├─ Código ad-hoc                 ├─ Código robusto
# MAGIC ├─ Paths hardcoded               ├─ Configuración
# MAGIC ├─ Sin versionado                ├─ Git
# MAGIC ├─ Datos locales                 ├─ Pipelines escalables
# MAGIC └─ Un solo modelo                └─ A/B testing
# MAGIC ```
# MAGIC
# MAGIC ### MLOps al Rescate
# MAGIC
# MAGIC **MLOps** cierra este gap aplicando prácticas de **DevOps** a ML:
# MAGIC
# MAGIC ```
# MAGIC 🔄 CI/CD para ML:
# MAGIC
# MAGIC Code → Train → Test → Deploy → Monitor → Retrain
# MAGIC   ↑                                          ↓
# MAGIC   └──────────────────────────────────────────┘
# MAGIC            (Continuous Improvement)
# MAGIC ```
# MAGIC
# MAGIC ### Componentes de MLOps
# MAGIC
# MAGIC #### 1️⃣ **Experiment Tracking** (MLflow)
# MAGIC
# MAGIC ```python
# MAGIC import mlflow
# MAGIC
# MAGIC with mlflow.start_run():
# MAGIC     # Track de TODO
# MAGIC     mlflow.log_params({"n_estimators": 100})
# MAGIC     mlflow.log_metrics({"accuracy": 0.95})
# MAGIC     mlflow.log_model(model, "model")
# MAGIC     mlflow.log_artifact("plot.png")
# MAGIC ```
# MAGIC
# MAGIC **Qué trackea:**
# MAGIC * Hiperparámetros
# MAGIC * Métricas (accuracy, RMSE, etc.)
# MAGIC * Modelos (archivos serializados)
# MAGIC * Artefactos (gráficos, datasets)
# MAGIC * Código (git commit)
# MAGIC * Entorno (librerías, versiones)
# MAGIC
# MAGIC #### 2️⃣ **Model Registry**
# MAGIC
# MAGIC ```
# MAGIC 📦 Model Registry (Versiones)
# MAGIC
# MAGIC ├── fraud_detection
# MAGIC │   ├── v1 (Staging)     ← Probando
# MAGIC │   ├── v2 (Production)  ← Sirviendo tráfico
# MAGIC │   └── v3 (Archived)    ← Descartado
# MAGIC │
# MAGIC └── churn_prediction
# MAGIC     └── v1 (Production)
# MAGIC ```
# MAGIC
# MAGIC **Stages:**
# MAGIC * **None**: Recién registrado
# MAGIC * **Staging**: En pruebas
# MAGIC * **Production**: Sirviendo predicciones
# MAGIC * **Archived**: Descartado
# MAGIC
# MAGIC #### 3️⃣ **Feature Store**
# MAGIC
# MAGIC ```python
# MAGIC from databricks.feature_store import FeatureStoreClient
# MAGIC
# MAGIC fs = FeatureStoreClient()
# MAGIC
# MAGIC # Crear feature table
# MAGIC fs.create_table(
# MAGIC     name="user_features",
# MAGIC     primary_keys=["user_id"],
# MAGIC     df=features_df,
# MAGIC     description="User behavioral features"
# MAGIC )
# MAGIC
# MAGIC # Usar en training
# MAGIC training_set = fs.create_training_set(
# MAGIC     df=labels_df,
# MAGIC     feature_lookups=[
# MAGIC         FeatureLookup(
# MAGIC             table_name="user_features",
# MAGIC             feature_names=["avg_session_time", "purchase_count"],
# MAGIC             lookup_key="user_id"
# MAGIC         )
# MAGIC     ]
# MAGIC )
# MAGIC ```
# MAGIC
# MAGIC **Ventajas:**
# MAGIC * ✅ **Reutilización**: Features usadas por múltiples modelos
# MAGIC * ✅ **Consistencia**: Mismo cálculo en training y serving
# MAGIC * ✅ **Trazabilidad**: Versionado de features
# MAGIC * ✅ **Freshness**: Features online (tiempo real)
# MAGIC
# MAGIC #### 4️⃣ **Model Serving**
# MAGIC
# MAGIC ```python
# MAGIC # Deploy automático desde Model Registry
# MAGIC import mlflow.deployments
# MAGIC
# MAGIC client = mlflow.deployments.get_deploy_client("databricks")
# MAGIC
# MAGIC endpoint = client.create_endpoint(
# MAGIC     name="fraud_detector",
# MAGIC     config={
# MAGIC         "served_models": [{
# MAGIC             "model_name": "fraud_detection",
# MAGIC             "model_version": "2",
# MAGIC             "workload_size": "Small",
# MAGIC             "scale_to_zero_enabled": True
# MAGIC         }]
# MAGIC     }
# MAGIC )
# MAGIC
# MAGIC # Invocar endpoint
# MAGIC predictions = client.predict(
# MAGIC     endpoint="fraud_detector",
# MAGIC     inputs={"dataframe_records": [{"amount": 150, "merchant": "Amazon"}]}
# MAGIC )
# MAGIC ```
# MAGIC
# MAGIC #### 5️⃣ **Monitoring**
# MAGIC
# MAGIC **Qué monitorear:**
# MAGIC
# MAGIC * **Performance del modelo**:
# MAGIC   - Accuracy, precision, recall en producción
# MAGIC   - Drift de métricas
# MAGIC   
# MAGIC * **Data Drift**: 
# MAGIC   - Distribución de features cambió
# MAGIC   - Ejemplo: Promedio de "amount" era $50, ahora es $200
# MAGIC   
# MAGIC * **Concept Drift**:
# MAGIC   - Relación X → Y cambió
# MAGIC   - Ejemplo: Antes "monto alto" → fraude, ahora no
# MAGIC   
# MAGIC * **Infraestructura**:
# MAGIC   - Latencia de predicciones
# MAGIC   - CPU/Memoria
# MAGIC   - Requests per second
# MAGIC
# MAGIC **Alertas automáticas:**
# MAGIC ```python
# MAGIC if accuracy_production < 0.8:
# MAGIC     send_alert("Model degradation detected")
# MAGIC     trigger_retraining()
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🛠️ Herramientas de AutoML

# COMMAND ----------

# DBTITLE 1,## Herramientas y Ecosistema
# MAGIC %md
# MAGIC ## 🛠️ Herramientas de AutoML
# MAGIC
# MAGIC ### Databricks AutoML ⭐ (Lo que usaremos)
# MAGIC
# MAGIC **Características:**
# MAGIC * ✅ **Integrado en la plataforma**: Sin instalación adicional
# MAGIC * ✅ **Genera notebooks explicativos**: Código reproducible
# MAGIC * ✅ **Optimización con Hyperopt**: Bayesian Optimization
# MAGIC * ✅ **MLflow tracking automático**: Todos los runs registrados
# MAGIC * ✅ **Feature Store integration**: Usa features existentes
# MAGIC * ✅ **Clasificación y Regresión**: Ambos soportados
# MAGIC
# MAGIC **Flujo:**
# MAGIC
# MAGIC ```python
# MAGIC from databricks import automl
# MAGIC
# MAGIC # 1️⃣ Ejecutar AutoML
# MAGIC summary = automl.classify(
# MAGIC     dataset=train_df,
# MAGIC     target_col="churn",
# MAGIC     primary_metric="f1",
# MAGIC     timeout_minutes=30,
# MAGIC     max_trials=50
# MAGIC )
# MAGIC
# MAGIC # 2️⃣ Ver mejores modelos
# MAGIC print(summary.best_trial.model_description)  # "LightGBM"
# MAGIC print(summary.best_trial.metrics)            # {"f1": 0.89, "auc": 0.92}
# MAGIC
# MAGIC # 3️⃣ Notebook generado
# MAGIC print(summary.best_trial.notebook_path)
# MAGIC # → Notebook con código completo del mejor modelo
# MAGIC ```
# MAGIC
# MAGIC **Modelos probados automáticamente:**
# MAGIC * Decision Tree
# MAGIC * Random Forest
# MAGIC * **LightGBM** ⭐ (usualmente el mejor)
# MAGIC * **XGBoost**
# MAGIC * Logistic Regression (clasificación)
# MAGIC * Linear Models (regresión)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Otras Herramientas Populares
# MAGIC
# MAGIC #### H2O.ai AutoML
# MAGIC
# MAGIC ```python
# MAGIC import h2o
# MAGIC from h2o.automl import H2OAutoML
# MAGIC
# MAGIC h2o.init()
# MAGIC aml = H2OAutoML(max_models=20, seed=1)
# MAGIC aml.train(x=features, y="target", training_frame=train)
# MAGIC
# MAGIC # Ver leaderboard
# MAGIC lb = aml.leaderboard
# MAGIC print(lb.head())
# MAGIC ```
# MAGIC
# MAGIC **Ventajas:**
# MAGIC * Open source
# MAGIC * Soporta muchos algoritmos (incluyendo Deep Learning)
# MAGIC * Excelente para tabular data
# MAGIC
# MAGIC #### Auto-sklearn
# MAGIC
# MAGIC ```python
# MAGIC import autosklearn.classification
# MAGIC
# MAGIC automl = autosklearn.classification.AutoSklearnClassifier(
# MAGIC     time_left_for_this_task=3600,  # 1 hora
# MAGIC     per_run_time_limit=300          # 5 min por modelo
# MAGIC )
# MAGIC automl.fit(X_train, y_train)
# MAGIC predictions = automl.predict(X_test)
# MAGIC ```
# MAGIC
# MAGIC **Ventajas:**
# MAGIC * Basado en scikit-learn
# MAGIC * Automatic ensemble construction
# MAGIC * Meta-learning (aprende de datasets pasados)
# MAGIC
# MAGIC #### TPOT (Tree-based Pipeline Optimization Tool)
# MAGIC
# MAGIC ```python
# MAGIC from tpot import TPOTClassifier
# MAGIC
# MAGIC tpot = TPOTClassifier(
# MAGIC     generations=5,
# MAGIC     population_size=20,
# MAGIC     verbosity=2
# MAGIC )
# MAGIC tpot.fit(X_train, y_train)
# MAGIC print(tpot.score(X_test, y_test))
# MAGIC
# MAGIC # Exportar pipeline
# MAGIC tpot.export('best_pipeline.py')
# MAGIC ```
# MAGIC
# MAGIC **Ventajas:**
# MAGIC * Usa algoritmos genéticos
# MAGIC * Optimiza pipelines completos (no solo modelos)
# MAGIC * Exporta código Python
# MAGIC
# MAGIC #### Google Cloud AutoML
# MAGIC
# MAGIC **Cloud-based, para:**
# MAGIC * AutoML Tables (tabular data)
# MAGIC * AutoML Vision (imágenes)
# MAGIC * AutoML Natural Language (texto)
# MAGIC * AutoML Video
# MAGIC
# MAGIC **Ventajas:**
# MAGIC * No requiere código
# MAGIC * Escalable
# MAGIC * Transfer learning automático
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## ⚖️ AutoML vs Manual: ¿Cuándo usar cada uno?
# MAGIC
# MAGIC ### ✅ Usar AutoML cuando:
# MAGIC
# MAGIC 1. **Necesitas un baseline rápido**
# MAGIC    - Proyecto nuevo
# MAGIC    - Exploración inicial
# MAGIC    - Prueba de concepto
# MAGIC
# MAGIC 2. **Tiempo limitado**
# MAGIC    - Deadline ajustado
# MAGIC    - Múltiples proyectos en paralelo
# MAGIC
# MAGIC 3. **Equipo con menos experiencia en ML**
# MAGIC    - Analistas de negocio
# MAGIC    - Ingenieros de datos
# MAGIC    - Product managers
# MAGIC
# MAGIC 4. **Dataset estándar (tabular)**
# MAGIC    - Clasificación binaria/multiclase
# MAGIC    - Regresión
# MAGIC    - No requiere arquitecturas custom
# MAGIC
# MAGIC 5. **Producción estándar**
# MAGIC    - Predicciones batch o API REST
# MAGIC    - Sin requisitos especiales de latencia
# MAGIC
# MAGIC ### 🔧 Usar Enfoque Manual cuando:
# MAGIC
# MAGIC 1. **Problema altamente especializado**
# MAGIC    - Computer Vision custom
# MAGIC    - NLP con arquitectura específica
# MAGIC    - Time Series con patrones únicos
# MAGIC    - Reinforcement Learning
# MAGIC
# MAGIC 2. **Necesitas control total**
# MAGIC    - Loss function custom
# MAGIC    - Arquitectura de red neuronal específica
# MAGIC    - Ensemble methods complejos
# MAGIC
# MAGIC 3. **Restricciones estrictas**
# MAGIC    - Latencia ultra-baja (<1ms)
# MAGIC    - Tamaño de modelo muy pequeño (edge devices)
# MAGIC    - Interpretabilidad total requerida
# MAGIC
# MAGIC 4. **Investigación y publicación**
# MAGIC    - Papers académicos
# MAGIC    - Métodos nuevos
# MAGIC    - Benchmarking riguroso
# MAGIC
# MAGIC 5. **Cuando AutoML falla**
# MAGIC    - Performance insuficiente
# MAGIC    - Overfitting severo
# MAGIC    - Data drift extremo
# MAGIC
# MAGIC ### 💡 Mejor Práctica: Enfoque Híbrido
# MAGIC
# MAGIC ```
# MAGIC 1. Ejecutar AutoML primero
# MAGIC    ↓
# MAGIC 2. Analizar notebook generado
# MAGIC    ↓  
# MAGIC 3. Entender qué funcionó
# MAGIC    ↓
# MAGIC 4. Iterar manualmente desde ahí
# MAGIC    ↓
# MAGIC 5. Feature engineering custom
# MAGIC    ↓
# MAGIC 6. Fine-tuning del mejor modelo
# MAGIC ```
# MAGIC
# MAGIC **Ejemplo real:**
# MAGIC
# MAGIC ```
# MAGIC AutoML baseline:  F1 = 0.87
# MAGIC     ↓
# MAGIC Análisis de features importantes
# MAGIC     ↓
# MAGIC Crear features nuevas (manual)
# MAGIC     ↓
# MAGIC Reentrenar con features nuevas
# MAGIC     ↓
# MAGIC Final:  F1 = 0.92  🎉
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 📊 Comparación: Manual vs AutoML
# MAGIC
# MAGIC | Aspecto | Manual | AutoML | MLOps |
# MAGIC |---------|--------|--------|-------|
# MAGIC | **Tiempo** | Días/Semanas | Horas | Continuo |
# MAGIC | **Experiencia requerida** | Alta | Baja/Media | Media/Alta |
# MAGIC | **Control** | Total | Limitado | Automatizado |
# MAGIC | **Interpretabilidad** | Alta | Media | Trazable |
# MAGIC | **Reproducibilidad** | Depende | Alta | Garantizada |
# MAGIC | **Escalabilidad** | Manual | Automática | Automática |
# MAGIC | **Costo** | Tiempo humano | Compute | Infraestructura |
# MAGIC | **Flexibilidad** | Máxima | Limitada | Configurable |
# MAGIC | **Best for** | Investigación | Prototipado rápido | Producción |
# MAGIC
# MAGIC ### Performance Típico
# MAGIC
# MAGIC ```
# MAGIC 📊 Benchmark en dataset tabular estándar:
# MAGIC
# MAGIC Manual (experto, 2 semanas):     F1 = 0.91
# MAGIC AutoML (30 minutos):             F1 = 0.89
# MAGIC Manual (principiante, 1 semana): F1 = 0.83
# MAGIC
# MAGIC → AutoML alcanza ~95% del performance del experto
# MAGIC    en 1/100 del tiempo
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🎯 El Stack Completo de MLOps en Databricks
# MAGIC
# MAGIC ```
# MAGIC ┌─────────────────────────────────────────────────┐
# MAGIC │              DATABRICKS PLATFORM                │
# MAGIC ├─────────────────────────────────────────────────┤
# MAGIC │                                                 │
# MAGIC │  📊 Data Engineering                            │
# MAGIC │     ├─ Delta Lake (Storage)                     │
# MAGIC │     ├─ Apache Spark (Processing)                │
# MAGIC │     └─ Workflows (Orchestration)                │
# MAGIC │                                                 │
# MAGIC │  🤖 Machine Learning                            │
# MAGIC │     ├─ AutoML (Automated Training) ⭐           │
# MAGIC │     ├─ Feature Store (Feature Management)       │
# MAGIC │     ├─ MLflow (Experiment Tracking)             │
# MAGIC │     └─ Model Registry (Model Versioning)        │
# MAGIC │                                                 │
# MAGIC │  🚀 Deployment                                  │
# MAGIC │     ├─ Model Serving (Real-time API)            │
# MAGIC │     ├─ Batch Inference (Spark Jobs)             │
# MAGIC │     └─ Model Monitoring (Drift Detection)       │
# MAGIC │                                                 │
# MAGIC │  🤝 Collaboration                               │
# MAGIC │     ├─ Notebooks (Interactive Dev)              │
# MAGIC │     ├─ Git Integration (Version Control)        │
# MAGIC │     ├─ Unity Catalog (Governance)               │
# MAGIC │     └─ Genie Code (AI Assistant) ⭐             │
# MAGIC │                                                 │
# MAGIC └─────────────────────────────────────────────────┘
# MAGIC ```
# MAGIC
# MAGIC ### Flujo End-to-End
# MAGIC
# MAGIC ```python
# MAGIC # 1. DATA ENGINEERING
# MAGIC raw_data = spark.read.table("bronze.transactions")
# MAGIC cleaned_data = clean_and_transform(raw_data)
# MAGIC cleaned_data.write.saveAsTable("silver.transactions")
# MAGIC
# MAGIC # 2. FEATURE ENGINEERING (Feature Store)
# MAGIC from databricks.feature_store import FeatureStoreClient
# MAGIC fs = FeatureStoreClient()
# MAGIC
# MAGIC features = compute_features(cleaned_data)
# MAGIC fs.create_table(
# MAGIC     name="features.user_behavior",
# MAGIC     primary_keys=["user_id"],
# MAGIC     df=features
# MAGIC )
# MAGIC
# MAGIC # 3. AUTOML TRAINING
# MAGIC from databricks import automl
# MAGIC
# MAGIC labels = spark.table("silver.churn_labels")
# MAGIC summary = automl.classify(
# MAGIC     dataset=labels,
# MAGIC     target_col="churned",
# MAGIC     feature_store_lookups=[
# MAGIC         {"table_name": "features.user_behavior", 
# MAGIC          "lookup_key": "user_id"}
# MAGIC     ],
# MAGIC     primary_metric="f1"
# MAGIC )
# MAGIC
# MAGIC # 4. MODEL REGISTRY
# MAGIC import mlflow
# MAGIC
# MAGIC model_uri = f"runs:/{summary.best_trial.mlflow_run_id}/model"
# MAGIC model_details = mlflow.register_model(model_uri, "churn_model")
# MAGIC
# MAGIC # 5. PROMOTE TO PRODUCTION
# MAGIC client = mlflow.tracking.MlflowClient()
# MAGIC client.transition_model_version_stage(
# MAGIC     name="churn_model",
# MAGIC     version=model_details.version,
# MAGIC     stage="Production"
# MAGIC )
# MAGIC
# MAGIC # 6. MODEL SERVING (Automatic)
# MAGIC # → Endpoint automáticamente disponible en:
# MAGIC # https://<workspace>.cloud.databricks.com/serving-endpoints/churn_model
# MAGIC
# MAGIC # 7. MONITORING
# MAGIC # → Métricas de drift, latencia, throughput automáticas
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 💡 Conclusiones

# COMMAND ----------

# DBTITLE 1,## Conclusiones y Mejores Prácticas
# MAGIC %md
# MAGIC ## 💡 Conclusiones y Mejores Prácticas
# MAGIC
# MAGIC ### La Realidad del ML en la Industria
# MAGIC
# MAGIC ```
# MAGIC 📊 Ciclo de vida del modelo en producción:
# MAGIC
# MAGIC 10% → Desarrollar modelo inicial
# MAGIC 20% → Optimizar y validar
# MAGIC 70% → Mantener en producción ⚠️
# MAGIC
# MAGIC → La mayoría del trabajo es MLOps, no desarrollo
# MAGIC ```
# MAGIC
# MAGIC ### Lecciones Clave
# MAGIC
# MAGIC #### 1️⃣ **AutoML no reemplaza a los Data Scientists**
# MAGIC
# MAGIC ✅ **AutoML es excelente para:**
# MAGIC * Baseline rápido
# MAGIC * Exploración inicial
# MAGIC * Producción de modelos "suficientemente buenos"
# MAGIC * Democratización del ML
# MAGIC
# MAGIC ❌ **AutoML no puede:**
# MAGIC * Definir el problema de negocio
# MAGIC * Decidir qué features son relevantes (requiere domain knowledge)
# MAGIC * Explicar por qué un modelo funciona
# MAGIC * Manejar edge cases complejos
# MAGIC * Innovar en arquitecturas
# MAGIC
# MAGIC **AutoML aumenta la productividad, no reemplaza el expertise.**
# MAGIC
# MAGIC #### 2️⃣ **MLOps es crucial para el éxito**
# MAGIC
# MAGIC ```
# MAGIC ❌ Sin MLOps:
# MAGIC
# MAGIC 📓 Jupyter notebook → 💾 Guardado localmente → 🤷 ¿Cómo lo uso?
# MAGIC
# MAGIC ✅ Con MLOps:
# MAGIC
# MAGIC 📓 Jupyter → 📦 MLflow → 🏭 Registry → 🚀 Serving → 📊 Monitor
# MAGIC                 ↓                                        ↓
# MAGIC          Trazabilidad                            Mejora continua
# MAGIC ```
# MAGIC
# MAGIC **Estadística clave:**
# MAGIC * 87% de los proyectos de ML nunca llegan a producción
# MAGIC * MLOps reduce este problema dramáticamente
# MAGIC
# MAGIC #### 3️⃣ **Feature Engineering sigue siendo el rey**
# MAGIC
# MAGIC ```
# MAGIC Impacto en performance:
# MAGIC
# MAGIC 📊 Mejor algoritmo:        +5%  accuracy
# MAGIC ⚙️  Mejor hyperparameters:  +3%  accuracy
# MAGIC 🔧 Mejores features:       +20% accuracy ⭐
# MAGIC ```
# MAGIC
# MAGIC **AutoML puede optimizar algoritmos, pero buenos features requieren:**
# MAGIC * Domain knowledge
# MAGIC * Creatividad
# MAGIC * Experimentación
# MAGIC
# MAGIC #### 4️⃣ **El modelo es solo una parte del sistema**
# MAGIC
# MAGIC ```
# MAGIC Sistema ML en producción:
# MAGIC
# MAGIC 📥 Data ingestion
# MAGIC     ↓
# MAGIC 🔄 Data validation
# MAGIC     ↓
# MAGIC 🧹 Data preprocessing  
# MAGIC     ↓
# MAGIC 🔧 Feature computation
# MAGIC     ↓
# MAGIC 🤖 Model inference  ← Solo 5% del código
# MAGIC     ↓
# MAGIC 📤 Predictions serving
# MAGIC     ↓
# MAGIC 📊 Monitoring & alerting
# MAGIC     ↓
# MAGIC 🔄 Retraining pipeline
# MAGIC ```
# MAGIC
# MAGIC #### 5️⃣ **Monitoreo es tan importante como entrenamiento**
# MAGIC
# MAGIC **Modelos se degradan con el tiempo:**
# MAGIC
# MAGIC ```
# MAGIC Day 1:  Accuracy = 0.95  ✅
# MAGIC Day 30: Accuracy = 0.93  ⚠️
# MAGIC Day 90: Accuracy = 0.87  🚨 Reentrenar
# MAGIC ```
# MAGIC
# MAGIC **Causas:**
# MAGIC * Data drift (distribución de X cambió)
# MAGIC * Concept drift (relación X→Y cambió)
# MAGIC * Cambios en el negocio
# MAGIC * Estacionalidad
# MAGIC
# MAGIC **Solución: Reentrenamiento automático**
# MAGIC
# MAGIC ```python
# MAGIC if drift_detected() or performance_drop():
# MAGIC     trigger_automl_retraining()
# MAGIC     validate_new_model()
# MAGIC     if new_model_better:
# MAGIC         promote_to_production()
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🚀 El Camino del ML Moderno
# MAGIC
# MAGIC ### Evolución Personal
# MAGIC
# MAGIC ```
# MAGIC 👨‍🎓 NIVEL 1: Principiante
# MAGIC ├─ Aprende algoritmos manualmente
# MAGIC ├─ Entiende matemáticas
# MAGIC └─ Experimenta con datasets toy
# MAGIC
# MAGIC 👨‍💻 NIVEL 2: Practicante
# MAGIC ├─ Usa AutoML para baselines
# MAGIC ├─ Feature engineering manual
# MAGIC ├─ Optimiza modelos
# MAGIC └─ Entiende trade-offs
# MAGIC
# MAGIC 👨‍🔬 NIVEL 3: Experto
# MAGIC ├─ Diseña arquitecturas custom
# MAGIC ├─ Combina AutoML + manual
# MAGIC ├─ Implementa MLOps completo
# MAGIC └─ Innova en métodos
# MAGIC
# MAGIC 🏭 NIVEL 4: ML Engineer
# MAGIC ├─ Sistemas end-to-end
# MAGIC ├─ Producción a escala
# MAGIC ├─ Monitoring & alerting
# MAGIC └─ Mejora continua
# MAGIC ```
# MAGIC
# MAGIC ### El Stack Moderno (2024)
# MAGIC
# MAGIC ```python
# MAGIC # Lo que aprenderemos en los notebooks prácticos:
# MAGIC
# MAGIC # 1. AutoML para desarrollo rápido
# MAGIC from databricks import automl
# MAGIC summary = automl.classify(data, target="churn")
# MAGIC
# MAGIC # 2. MLflow para tracking
# MAGIC import mlflow
# MAGIC with mlflow.start_run():
# MAGIC     mlflow.log_metrics({"accuracy": 0.95})
# MAGIC
# MAGIC # 3. Feature Store para reutilización
# MAGIC from databricks.feature_store import FeatureStoreClient
# MAGIC fs = FeatureStoreClient()
# MAGIC
# MAGIC # 4. Model Registry para versioning
# MAGIC mlflow.register_model(model_uri, "my_model")
# MAGIC
# MAGIC # 5. Model Serving para deployment
# MAGIC # Endpoint automático disponible
# MAGIC
# MAGIC # 6. Monitoring para producción
# MAGIC # Drift detection automático
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 📚 Recursos y Referencias
# MAGIC
# MAGIC ### Documentación Oficial
# MAGIC
# MAGIC * **Databricks AutoML**: [docs.databricks.com/machine-learning/automl](https://docs.databricks.com/machine-learning/automl/index.html)
# MAGIC * **MLflow**: [mlflow.org/docs/latest](https://mlflow.org/docs/latest/index.html)
# MAGIC * **Feature Store**: [docs.databricks.com/machine-learning/feature-store](https://docs.databricks.com/machine-learning/feature-store/index.html)
# MAGIC * **Model Serving**: [docs.databricks.com/machine-learning/model-serving](https://docs.databricks.com/machine-learning/model-serving/index.html)
# MAGIC
# MAGIC ### Papers Importantes
# MAGIC
# MAGIC * **AutoML Survey**: "AutoML: A Survey of the State-of-the-Art" (2020)
# MAGIC * **MLOps**: "Hidden Technical Debt in Machine Learning Systems" (Google, 2015)
# MAGIC * **Hyperparameter Optimization**: "Algorithms for Hyper-Parameter Optimization" (Bergstra et al., 2011)
# MAGIC
# MAGIC ### Cursos Recomendados
# MAGIC
# MAGIC * **MLOps Specialization** (DeepLearning.AI + Coursera)
# MAGIC * **Machine Learning Engineering for Production** (Andrew Ng)
# MAGIC * **Databricks Academy** - ML in Production
# MAGIC
# MAGIC ### Libros
# MAGIC
# MAGIC * **"Designing Machine Learning Systems"** - Chip Huyen
# MAGIC * **"Machine Learning Engineering"** - Andriy Burkov
# MAGIC * **"Introducing MLOps"** - Mark Treveil et al.
# MAGIC * **"Building Machine Learning Powered Applications"** - Emmanuel Ameisen
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🎯 Próximos Pasos
# MAGIC
# MAGIC **En los notebooks prácticos veremos:**
# MAGIC
# MAGIC 1. **Databricks AutoML en acción**
# MAGIC    - Clasificación (churn prediction)
# MAGIC    - Regresión (price prediction)
# MAGIC    - Análisis de notebooks generados
# MAGIC
# MAGIC 2. **Genie Code como asistente**
# MAGIC    - Generar pipelines completos
# MAGIC    - Optimizar modelos
# MAGIC    - Debugging asistido
# MAGIC
# MAGIC 3. **MLflow end-to-end**
# MAGIC    - Experiment tracking
# MAGIC    - Model registry
# MAGIC    - Deployment
# MAGIC
# MAGIC 4. **Feature Store**
# MAGIC    - Crear features reutilizables
# MAGIC    - Training con Feature Store
# MAGIC    - Online serving
# MAGIC
# MAGIC 5. **Comparación Manual vs AutoML**
# MAGIC    - Mismo problema
# MAGIC    - Dos enfoques
# MAGIC    - Análisis de trade-offs
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 💭 Reflexión Final
# MAGIC
# MAGIC > **"AutoML democratiza el Machine Learning, MLOps lo hace sostenible en producción, y el expertise humano sigue siendo insustituible para problemas complejos."**
# MAGIC
# MAGIC La combinación de:
# MAGIC * 🤖 **AutoML** (velocidad)
# MAGIC * 🏭 **MLOps** (robustez)
# MAGIC * 👨‍🔬 **Expertise humano** (innovación)
# MAGIC
# MAGIC ...es el futuro del Machine Learning en la industria.
# MAGIC
# MAGIC **¡Pasemos a la práctica!** 🚀

# COMMAND ----------

