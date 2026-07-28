# Databricks notebook source
# DBTITLE 1,# MLflow - Experiment Tracking
# MAGIC %md
# MAGIC # MLflow Experiment Tracking y Model Registry
# MAGIC
# MAGIC ## 🎯 Objetivo
# MAGIC Dominar **MLflow** para tracking, registry, deployment y monitoreo.
# MAGIC
# MAGIC ### Workflow: Track → Compare → Register → Deploy → Monitor

# COMMAND ----------

# DBTITLE 1,Setup y tracking
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

mlflow.set_experiment("/Users/dba@uda.edu.ar/ml_tracking_demo")

np.random.seed(42)
n = 1000
X = np.random.randn(n, 10)
y = (X[:, 0] + X[:, 1] > 0).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Entrenar 3 modelos
configs = [
    {'n_estimators': 50, 'max_depth': 5},
    {'n_estimators': 100, 'max_depth': 10},
    {'n_estimators': 200, 'max_depth': 15}
]

for i, config in enumerate(configs, 1):
    with mlflow.start_run(run_name=f"rf_model_{i}"):
        mlflow.log_params(config)
        model = RandomForestClassifier(**config, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        mlflow.log_metrics({'accuracy': accuracy, 'f1_score': f1_score(y_test, y_pred)})
        mlflow.sklearn.log_model(model, "model")
        
print("✅ 3 runs registrados en MLflow UI")

# COMMAND ----------

# DBTITLE 1,Model Registry
print("""
📑 MODEL REGISTRY

1. Registrar:
   mlflow.register_model(model_uri, name="my_model")

2. Transiciones:
   None → Staging → Production → Archived

3. Cargar:
   model = mlflow.pyfunc.load_model("models:/my_model/Production")

✅ Versionado + Control de estados
""")

# COMMAND ----------

# DBTITLE 1,Deployment options
# MAGIC %md
# MAGIC ## 🚀 Model Serving
# MAGIC
# MAGIC ### 1. Batch Inference
# MAGIC ```python
# MAGIC model = mlflow.pyfunc.load_model("models:/my_model/Production")
# MAGIC df = spark.table("customer_data")
# MAGIC predictions = model.predict(df.toPandas())
# MAGIC ```
# MAGIC
# MAGIC ### 2. Real-time Endpoint
# MAGIC ```python
# MAGIC endpoint_url = "https://<workspace>/serving-endpoints/my_model/invocations"
# MAGIC response = requests.post(endpoint_url, json=data, headers=headers)
# MAGIC ```
# MAGIC
# MAGIC ### 3. Streaming
# MAGIC ```python
# MAGIC model_udf = mlflow.pyfunc.spark_udf(spark, "models:/my_model/Production")
# MAGIC stream.withColumn("prediction", model_udf(...))
# MAGIC ```

# COMMAND ----------

# DBTITLE 1,Monitoreo y drift
# MAGIC %md
# MAGIC ## 📊 Monitoreo
# MAGIC
# MAGIC ### Model Drift
# MAGIC * **Data drift**: Distribución de X cambia
# MAGIC * **Concept drift**: Relación X→Y cambia
# MAGIC
# MAGIC ### Detectar
# MAGIC ```python
# MAGIC from scipy.stats import ks_2samp
# MAGIC statistic, p_value = ks_2samp(train_feature, prod_feature)
# MAGIC if p_value < 0.05:
# MAGIC     print("⚠️  Drift detectado - Reentrenar")
# MAGIC ```
# MAGIC
# MAGIC ### Retraining Pipeline
# MAGIC 1. Monitorear métricas
# MAGIC 2. Detectar degradación
# MAGIC 3. Recolectar datos nuevos
# MAGIC 4. Reentrenar modelo
# MAGIC 5. A/B test
# MAGIC 6. Promover a Production

# COMMAND ----------

# DBTITLE 1,Feature Store
# MAGIC %md
# MAGIC ## 💾 Feature Store
# MAGIC
# MAGIC **Repositorio centralizado de features**
# MAGIC
# MAGIC ### Ventajas
# MAGIC * Reutilización entre equipos
# MAGIC * Consistencia train/prod
# MAGIC * Versionado
# MAGIC * Lineage
# MAGIC
# MAGIC ### Ejemplo
# MAGIC ```python
# MAGIC from databricks import feature_store
# MAGIC fs = feature_store.FeatureStoreClient()
# MAGIC
# MAGIC fs.create_table(
# MAGIC     name="ml.customer_features",
# MAGIC     primary_keys=["customer_id"],
# MAGIC     df=features_df
# MAGIC )
# MAGIC
# MAGIC training_set = fs.create_training_set(
# MAGIC     df=labels_df,
# MAGIC     feature_lookups=[...],
# MAGIC     label="churn"
# MAGIC )
# MAGIC
# MAGIC fs.log_model(model, training_set=training_set)
# MAGIC ```

# COMMAND ----------

# DBTITLE 1,## Conclusiones
# MAGIC %md
# MAGIC ## 📝 Conclusiones
# MAGIC
# MAGIC ### Key Takeaways
# MAGIC
# MAGIC 1. **MLflow Tracking**: Parámetros, métricas, modelos
# MAGIC 2. **Model Registry**: Versionado + Staging → Production
# MAGIC 3. **Deployment**: Batch, Real-time, Streaming
# MAGIC 4. **Monitoreo**: Data drift + Retraining automático
# MAGIC 5. **Feature Store**: Reutilización + Consistencia
# MAGIC
# MAGIC ### 🚀 Próximos Pasos
# MAGIC
# MAGIC * CI/CD para ML
# MAGIC * Monitoring dashboard
# MAGIC * Multi-model A/B testing
# MAGIC
# MAGIC ## 🎉 ¡Felicidades!
# MAGIC
# MAGIC **Has completado AutoML y MLOps.**
# MAGIC Listo para **producción profesional**. 🚀

# COMMAND ----------

