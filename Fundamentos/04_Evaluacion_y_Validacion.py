# Databricks notebook source
# DBTITLE 1,# Evaluación y Validación de Modelos
# MAGIC %md
# MAGIC # Evaluación y Validación de Modelos de Machine Learning
# MAGIC
# MAGIC ## 🎯 Objetivo
# MAGIC
# MAGIC Aprender a **evaluar correctamente** modelos de ML y detectar problemas como overfitting y underfitting.
# MAGIC
# MAGIC ### 📌 Tema Central
# MAGIC
# MAGIC > **"Un modelo es tan bueno como su evaluación."**
# MAGIC
# MAGIC No basta con entrenar un modelo - necesitamos saber:
# MAGIC * ¿Qué tan bien funciona?
# MAGIC * ¿Funcionará con datos nuevos?
# MAGIC * ¿Qué tipo de errores comete?
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 📚 Contenido
# MAGIC
# MAGIC 1. **Métricas de Evaluación**
# MAGIC    - Clasificación: Accuracy, Precision, Recall, F1, ROC-AUC
# MAGIC    - Regresión: MSE, RMSE, MAE, R²
# MAGIC
# MAGIC 2. **Validación**
# MAGIC    - Train/Test Split
# MAGIC    - Validación Cruzada (K-Fold)
# MAGIC    - Estratificación
# MAGIC
# MAGIC 3. **Problemas Comunes**
# MAGIC    - Overfitting (Sobreajuste)
# MAGIC    - Underfitting (Subajuste)
# MAGIC    - Bias-Variance Tradeoff
# MAGIC
# MAGIC 4. **Curvas de Aprendizaje**
# MAGIC    - Diagnóstico de problemas
# MAGIC    - Soluciones

# COMMAND ----------

# DBTITLE 1,## Parte 1: Métricas de Clasificación
# MAGIC %md
# MAGIC ## Parte 1: Métricas de Clasificación
# MAGIC
# MAGIC ### 🎯 Matriz de Confusión
# MAGIC
# MAGIC Base de todas las métricas de clasificación:
# MAGIC
# MAGIC ```
# MAGIC                     Predicción
# MAGIC                 Positivo  Negativo
# MAGIC Real  Positivo    TP        FN
# MAGIC       Negativo    FP        TN
# MAGIC ```
# MAGIC
# MAGIC * **TP (True Positive)**: Predijo Positivo, era Positivo ✅
# MAGIC * **TN (True Negative)**: Predijo Negativo, era Negativo ✅
# MAGIC * **FP (False Positive)**: Predijo Positivo, era Negativo ❌ (Error Tipo I)
# MAGIC * **FN (False Negative)**: Predijo Negativo, era Positivo ❌ (Error Tipo II)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 📊 Métricas Principales
# MAGIC
# MAGIC #### 1️⃣ **Accuracy (Exactitud)**
# MAGIC
# MAGIC $$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$
# MAGIC
# MAGIC * **Interpretación**: % de predicciones correctas
# MAGIC * **Cuándo usar**: Clases balanceadas
# MAGIC * **Problema**: Engañosa con clases desbalanceadas
# MAGIC
# MAGIC **Ejemplo**: Dataset con 95% clase negativa
# MAGIC * Modelo dummy que predice siempre "Negativo" → 95% accuracy ❌
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC #### 2️⃣ **Precision (Precisión)**
# MAGIC
# MAGIC $$\text{Precision} = \frac{TP}{TP + FP}$$
# MAGIC
# MAGIC * **Interpretación**: De los que predije como Positivos, ¿cuántos eran realmente Positivos?
# MAGIC * **Cuándo usar**: Minimizar **Falsos Positivos** es crítico
# MAGIC * **Ejemplo**: Detección de spam (no queremos marcar emails legítimos como spam)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC #### 3️⃣ **Recall (Sensibilidad / Exhaustividad)**
# MAGIC
# MAGIC $$\text{Recall} = \frac{TP}{TP + FN}$$
# MAGIC
# MAGIC * **Interpretación**: De los realmente Positivos, ¿cuántos detecté?
# MAGIC * **Cuándo usar**: Minimizar **Falsos Negativos** es crítico
# MAGIC * **Ejemplo**: Diagnóstico de cáncer (no queremos perder ningún caso positivo)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC #### 4️⃣ **F1-Score**
# MAGIC
# MAGIC $$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$
# MAGIC
# MAGIC * **Interpretación**: Media armónica de Precision y Recall
# MAGIC * **Cuándo usar**: Balance entre Precision y Recall, clases desbalanceadas
# MAGIC * **Rango**: [0, 1], 1 es perfecto
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC #### 5️⃣ **ROC-AUC (Area Under Curve)**
# MAGIC
# MAGIC * **ROC Curve**: True Positive Rate vs False Positive Rate
# MAGIC * **AUC**: Área bajo la curva ROC
# MAGIC * **Interpretación**: Probabilidad de que el modelo rankee un positivo aleatorio más alto que un negativo aleatorio
# MAGIC * **Rango**: [0.5, 1.0]
# MAGIC   - 0.5 = Azar
# MAGIC   - 1.0 = Perfecto

# COMMAND ----------

# DBTITLE 1,Ejemplo: Métricas de clasificación
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd

# Dataset de ejemplo: Detección de fraude (desbalanceado)
np.random.seed(42)
n = 1000
X = np.random.randn(n, 5)
y = np.random.binomial(1, 0.05, n)  # 5% fraudes

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

print("╔════════════════════════════════════════════════════════════════════════╗")
print("║              MÉTRICAS DE CLASIFICACIÓN - EJEMPLO                       ║")
print("╚════════════════════════════════════════════════════════════════════════╝")

print(f"\n📊 Clase 0 (No Fraude): {(y_test == 0).sum()} ({(y_test == 0).sum()/len(y_test)*100:.1f}%)")
print(f"   Clase 1 (Fraude): {(y_test == 1).sum()} ({(y_test == 1).sum()/len(y_test)*100:.1f}%)")

cm = confusion_matrix(y_test, y_pred)
print("\n🎯 MATRIZ DE CONFUSIÓN:")
print(f"   TN={cm[0,0]} FP={cm[0,1]}")
print(f"   FN={cm[1,0]} TP={cm[1,1]}")

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"\n📈 MÉTRICAS:")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")

# COMMAND ----------

# DBTITLE 1,## Parte 2: Validación Cruzada
# MAGIC %md
# MAGIC ## Parte 2: Validación Cruzada (K-Fold)
# MAGIC
# MAGIC ### 🎯 El Problema
# MAGIC
# MAGIC **Train/Test Split simple**: El resultado puede depender del split aleatorio
# MAGIC
# MAGIC ### 💡 Solución: K-Fold Cross Validation
# MAGIC
# MAGIC ```
# MAGIC K = 5 folds
# MAGIC
# MAGIC Fold 1: [Test][Train][Train][Train][Train]
# MAGIC Fold 2: [Train][Test][Train][Train][Train]
# MAGIC Fold 3: [Train][Train][Test][Train][Train]
# MAGIC Fold 4: [Train][Train][Train][Test][Train]
# MAGIC Fold 5: [Train][Train][Train][Train][Test]
# MAGIC          ↓
# MAGIC Promedio de 5 scores → Estimación más robusta
# MAGIC ```
# MAGIC
# MAGIC **Ventajas**:
# MAGIC * ✅ Usa todos los datos para entrenar y evaluar
# MAGIC * ✅ Estimación más robusta del performance
# MAGIC * ✅ Reduce varianza del score
# MAGIC
# MAGIC **Stratified K-Fold**: Mantiene proporción de clases en cada fold (importante para clases desbalanceadas)

# COMMAND ----------

# DBTITLE 1,Ejemplo: Validación cruzada
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

print("╔════════════════════════════════════════════════════════════════════════╗")
print("║                VALIDACIÓN CRUZADA - EJEMPLO                            ║")
print("╚════════════════════════════════════════════════════════════════════════╝")

model_cv = DecisionTreeClassifier(max_depth=5, random_state=42)

# K-Fold Cross Validation
scores_cv = cross_val_score(model_cv, X, y, cv=5, scoring='accuracy')

print(f"\n📊 K-FOLD (k=5) - Accuracy:")
for i, score in enumerate(scores_cv, 1):
    print(f"  Fold {i}: {score:.4f}")

print(f"\nPromedio: {scores_cv.mean():.4f} (± {scores_cv.std():.4f})")

# Stratified K-Fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores_strat = cross_val_score(model_cv, X, y, cv=skf, scoring='f1')

print(f"\n📊 STRATIFIED K-FOLD (k=5) - F1:")
for i, score in enumerate(scores_strat, 1):
    print(f"  Fold {i}: {score:.4f}")

print(f"\nPromedio: {scores_strat.mean():.4f} (± {scores_strat.std():.4f})")
print("\n✅ Stratified mantiene proporción de clases en cada fold")

# COMMAND ----------

# DBTITLE 1,## Parte 3: Overfitting y Underfitting
# MAGIC %md
# MAGIC ## Parte 3: Overfitting y Underfitting
# MAGIC
# MAGIC ### 📉 **Underfitting (Subajuste)**
# MAGIC
# MAGIC **Definición**: Modelo demasiado simple
# MAGIC
# MAGIC **Síntomas**:
# MAGIC * ❌ Train error alto
# MAGIC * ❌ Test error alto
# MAGIC * ❌ Train ≈ Test (ambos malos)
# MAGIC
# MAGIC **Soluciones**:
# MAGIC * ✅ Modelo más complejo
# MAGIC * ✅ Más features
# MAGIC * ✅ Reducir regularización
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 📈 **Overfitting (Sobreajuste)**
# MAGIC
# MAGIC **Definición**: Modelo memoriza el training set (incluyendo ruido)
# MAGIC
# MAGIC **Síntomas**:
# MAGIC * ❌ Train error bajo
# MAGIC * ❌ Test error alto
# MAGIC * ❌ Gran gap entre train y test
# MAGIC
# MAGIC **Soluciones**:
# MAGIC * ✅ Más datos de entrenamiento
# MAGIC * ✅ Regularización (L1, L2)
# MAGIC * ✅ Early stopping
# MAGIC * ✅ Reducir complejidad
# MAGIC * ✅ Feature selection
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### ⚖️ **Bias-Variance Tradeoff**
# MAGIC
# MAGIC $$\text{Error Total} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Error}$$
# MAGIC
# MAGIC * **Bias**: Error por simplicidad (alto → underfitting)
# MAGIC * **Variance**: Sensibilidad a variaciones en training data (alta → overfitting)
# MAGIC * **Objetivo**: Minimizar ambos

# COMMAND ----------

# DBTITLE 1,Ejemplo: Diagnóstico de overfitting
from sklearn.tree import DecisionTreeRegressor

print("╔════════════════════════════════════════════════════════════════════════╗")
print("║            OVERFITTING VS UNDERFITTING - DIAGNÓSTICO                   ║")
print("╚════════════════════════════════════════════════════════════════════════╝")

# Dataset sintético
np.random.seed(42)
X_curve = np.sort(np.random.rand(100, 1) * 10, axis=0)
y_curve = np.sin(X_curve).ravel() + np.random.randn(100) * 0.5

X_tr, X_te, y_tr, y_te = train_test_split(X_curve, y_curve, test_size=0.3, random_state=42)

models = {
    'Underfitting (depth=1)': DecisionTreeRegressor(max_depth=1),
    'Just Right (depth=5)': DecisionTreeRegressor(max_depth=5),
    'Overfitting (depth=20)': DecisionTreeRegressor(max_depth=20)
}

print("\n📊 COMPARACIÓN:\n")

for name, model in models.items():
    model.fit(X_tr, y_tr)
    train_error = ((model.predict(X_tr) - y_tr) ** 2).mean()
    test_error = ((model.predict(X_te) - y_te) ** 2).mean()
    gap = test_error - train_error
    
    print(f"{name}:")
    print(f"  Train MSE: {train_error:.4f}")
    print(f"  Test MSE:  {test_error:.4f}")
    print(f"  Gap:       {gap:.4f}")
    
    if train_error > 0.5 and gap < 0.1:
        print("  ⚠️  UNDERFITTING: Ambos errores altos")
    elif gap > 0.5:
        print("  ⚠️  OVERFITTING: Gran gap train-test")
    else:
        print("  ✅ BALANCED")
    print()

# COMMAND ----------

# DBTITLE 1,## 📝 Conclusiones
# MAGIC %md
# MAGIC ## 📝 Conclusiones y Mejores Prácticas
# MAGIC
# MAGIC ### 🎯 Key Takeaways
# MAGIC
# MAGIC 1. **La métrica importa**
# MAGIC    - Clasificación: F1 para desbalance, ROC-AUC para ranking
# MAGIC    - Regresión: RMSE (penaliza outliers), MAE (robusto)
# MAGIC    - Nunca solo Accuracy en clases desbalanceadas
# MAGIC
# MAGIC 2. **Validación es crítica**
# MAGIC    - Siempre usar train/test split
# MAGIC    - K-Fold CV para estimación robusta
# MAGIC    - Stratified para clases desbalanceadas
# MAGIC
# MAGIC 3. **Overfitting es el enemigo #1**
# MAGIC    - Monitorear train vs test error
# MAGIC    - Gap grande → Regularizar
# MAGIC    - Más datos > Modelo complejo
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### ✅ Checklist de Evaluación
# MAGIC
# MAGIC ☑️ Split estratificado train/test
# MAGIC ☑️ Métricas apropiadas (F1/ROC-AUC para clasificación, R²/RMSE para regresión)
# MAGIC ☑️ Validación cruzada (K-Fold, k=5 o k=10)
# MAGIC ☑️ Diagnóstico: curvas de aprendizaje, train vs test
# MAGIC ☑️ No overfitting: gap pequeño train-test
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🚀 Próximos Pasos
# MAGIC
# MAGIC **Fundamentos completos:**
# MAGIC * ✅ Introducción a ML
# MAGIC * ✅ Matemáticas esenciales
# MAGIC * ✅ Preprocesamiento y Feature Engineering
# MAGIC * ✅ Evaluación y Validación
# MAGIC
# MAGIC **Siguiente: Aprendizaje Supervisado** - Algoritmos específicos
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🎉 ¡Felicidades!
# MAGIC
# MAGIC Ahora puedes **evaluar correctamente** cualquier modelo de ML. 🚀

# COMMAND ----------

