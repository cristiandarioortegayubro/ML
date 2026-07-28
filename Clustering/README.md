# Clustering - Segmentación de Clientes

## 📂 Contenido

Esta carpeta contiene ejemplos de **Machine Learning no supervisado: Clustering** usando PySpark ML.

### Notebooks:

1. **KMeans_Clustering.py**
   - **Algoritmo**: K-Means Clustering
   - **Problema**: Segmentación de clientes de e-commerce
   - **Dataset**: 5,000 clientes con variables RFM (Recency, Frequency, Monetary)
   - **Métrica**: WSSSE, Silhouette Score
   - **Caso de uso**: Diseñar estrategias de marketing personalizadas por segmento
   - **Segmentos**: VIP, Regulares, En Riesgo, Nuevos/Ocasionales

## 🎯 Objetivo

Aprender a descubrir **grupos naturales** en los datos sin etiquetas previas, usando algoritmos de clustering.

## 📊 Concepto: Clustering

**Clustering** (agrupamiento) es una técnica de **aprendizaje no supervisado** que agrupa datos similares sin necesidad de etiquetas previas.

### Diferencia con Aprendizaje Supervisado:

* **Supervisado** (Clasificación/Regresión):
  * Requiere datos etiquetados (ej: Churn Sí/No)
  * Predice etiquetas para datos nuevos
  * Entrenamiento guiado

* **No Supervisado** (Clustering):
  * NO requiere etiquetas
  * Descubre patrones ocultos
  * Agrupa datos similares automáticamente

### Algoritmos de Clustering:

* **K-Means**: Agrupa en K clusters esféricos (más usado)
* **DBSCAN**: Detecta clusters de forma arbitraria y outliers
* **Hierarchical Clustering**: Crea dendrograma jerárquico
* **Gaussian Mixture Models**: Asigna probabilidades de pertenencia

### K-Means en detalle:

**Funcionamiento:**
1. Elegir K (número de clusters)
2. Inicializar K centroides aleatoriamente
3. Asignar cada punto al centroide más cercano (distancia euclídea)
4. Recalcular centroides (media de puntos en cada cluster)
5. Repetir hasta convergencia

**Ventajas:**
* Simple y rápido
* Escala bien con grandes datasets
* Fácil de interpretar

**Limitaciones:**
* Requiere especificar K a priori (usar Elbow Method)
* Sensible a escala (requiere normalización)
* Asume clusters esféricos
* Sensible a inicialización (usar seed)

## 🚀 Cómo usar este notebook

1. **Abrir en Databricks**:
   - Navega a `ML/Clustering`
   - Abre `KMeans_Clustering`

2. **Ejecutar**:
   - El notebook incluye **Elbow Method** para determinar K óptimo
   - Entrena K-Means con K=4
   - Perfila cada segmento con estadísticas

3. **Interpretar**:
   - **Cluster 0 (VIP)**: Alta frecuencia, alto gasto → Retener con programa VIP
   - **Cluster 1 (Regulares)**: Actividad media → Up-sell a VIP
   - **Cluster 2 (En Riesgo)**: Inactivos → Campaña de reactivación
   - **Cluster 3 (Nuevos)**: Baja actividad → Onboarding

4. **Experimentar**:
   - Prueba con diferentes valores de K
   - Añade nuevas features (categoría de producto, canal preferido)
   - Aplica a tus propios datos de clientes

## 📚 Recursos

* [PySpark ML Clustering](https://spark.apache.org/docs/latest/ml-clustering.html)
* [K-Means Algorithm](https://spark.apache.org/docs/latest/ml-clustering.html#k-means)
* [RFM Analysis](https://en.wikipedia.org/wiki/RFM_(market_research))
* [Customer Segmentation Guide](https://www.databricks.com/glossary/customer-segmentation)

## 💼 Aplicaciones reales

* **Marketing**: Segmentación de clientes para campañas personalizadas
* **Retail**: Grupos de productos para recomendaciones
* **Fraud Detection**: Detección de transacciones anómalas
* **Image Compression**: Reducción de colores (paleta limitada)
* **Document Clustering**: Agrupación temática de textos
* **Anomaly Detection**: Identificación de outliers

## 📊 Estrategias de Marketing por Segmento

### Cluster 0 (VIP) - 15% de clientes, 60% de revenue
* Programa de lealtad exclusivo
* Acceso anticipado a nuevos productos
* Atención personalizada

### Cluster 1 (Regulares) - 40% de clientes
* Up-sell a productos premium
* Programas de puntos
* Email marketing personalizado

### Cluster 2 (En Riesgo) - 25% de clientes
* Campaña de win-back con descuentos
* Encuesta de satisfacción
* Ofertas limitadas

### Cluster 3 (Nuevos) - 20% de clientes
* Onboarding educativo
* Descuento de bienvenida
* Incentivo para segunda compra

---

**¡Felicitaciones!** Has completado los tres tipos principales de Machine Learning: Clasificación, Regresión y Clustering.