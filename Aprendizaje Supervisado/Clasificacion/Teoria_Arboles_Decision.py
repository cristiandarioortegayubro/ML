# Databricks notebook source
# DBTITLE 1,Teoría de Árboles de Decisión
# MAGIC %md
# MAGIC # Teoría de Árboles de Decisión
# MAGIC
# MAGIC ## 1. Introducción
# MAGIC
# MAGIC Los **Árboles de Decisión** son algoritmos de aprendizaje supervisado que modelan decisiones mediante una estructura de árbol jerárquica. Son útiles tanto para clasificación como para regresión.
# MAGIC
# MAGIC ### Estructura de un Árbol de Decisión
# MAGIC
# MAGIC ```
# MAGIC                     [Nodo Raíz]
# MAGIC                    (feature X1)
# MAGIC                     /        \\
# MAGIC                    /          \\
# MAGIC               X1 < 5         X1 >= 5
# MAGIC               /                  \\
# MAGIC         [Nodo Interno]      [Nodo Interno]
# MAGIC          (feature X2)        (feature X3)
# MAGIC           /      \\             /      \\
# MAGIC         /         \\           /         \\
# MAGIC     [Hoja:A]  [Hoja:B]  [Hoja:C]  [Hoja:D]
# MAGIC ```
# MAGIC
# MAGIC **Componentes:**
# MAGIC - **Nodo Raíz**: Primer nodo, contiene todos los datos
# MAGIC - **Nodos Internos**: Decisiones basadas en características
# MAGIC - **Ramas**: Resultados de las decisiones (test de condición)
# MAGIC - **Nodos Hoja**: Decisión final (clase o valor)
# MAGIC
# MAGIC ### Definición Formal
# MAGIC
# MAGIC Un árbol de decisión es una función $h: \\mathbb{X} \\rightarrow \\mathbb{Y}$ que particiona recursivamente el espacio de entrada $\\mathbb{X}$ mediante reglas de decisión:
# MAGIC
# MAGIC $$h(\\mathbf{x}) = \\begin{cases}
# MAGIC y_1 & \\text{if } \\mathbf{x} \\in R_1 \\\\
# MAGIC y_2 & \\text{if } \\mathbf{x} \\in R_2 \\\\
# MAGIC \\vdots \\\\
# MAGIC y_k & \\text{if } \\mathbf{x} \\in R_k
# MAGIC \\end{cases}$$
# MAGIC
# MAGIC Donde $R_1, R_2, ..., R_k$ son regiones del espacio de características y $y_i$ es la predicción para la región $R_i$.

# COMMAND ----------

# DBTITLE 1,Algoritmo de Construcción
# MAGIC %md
# MAGIC ## 2. Algoritmo de Construcción del Árbol
# MAGIC
# MAGIC ### Algoritmo ID3 (Iterative Dichotomiser 3)
# MAGIC
# MAGIC **Pseudocódigo:**
# MAGIC
# MAGIC ```python
# MAGIC función ID3(ejemplos, atributos, clase_objetivo):
# MAGIC     if todos_ejemplos_misma_clase(ejemplos):
# MAGIC         return crear_hoja(clase_mayoritaria)
# MAGIC     
# MAGIC     if len(atributos) == 0:
# MAGIC         return crear_hoja(clase_mayoritaria(ejemplos))
# MAGIC     
# MAGIC     mejor_atributo = max(atributos, key=lambda a: ganancia_informacion(a, ejemplos))
# MAGIC     nodo = Nodo(mejor_atributo)
# MAGIC     
# MAGIC     for valor in valores(mejor_atributo):
# MAGIC         ejemplos_v = filtrar(ejemplos, mejor_atributo == valor)
# MAGIC         if len(ejemplos_v) == 0:
# MAGIC             añadir_hoja(nodo, clase_mayoritaria(ejemplos))
# MAGIC         else:
# MAGIC             atributos_restantes = atributos - {mejor_atributo}
# MAGIC             subárbol = ID3(ejemplos_v, atributos_restantes, clase_objetivo)
# MAGIC             añadir_rama(nodo, valor, subárbol)
# MAGIC     return nodo
# MAGIC ```

# COMMAND ----------

# DBTITLE 1,Entropía y Ganancia
# MAGIC %md
# MAGIC ## 3. Entropía y Ganancia de Información
# MAGIC
# MAGIC ### 3.1 Entropía
# MAGIC
# MAGIC Mide la **impureza** de un conjunto de datos:
# MAGIC
# MAGIC $$H(S) = -\\sum_{i=1}^{c} p_i \\log_2(p_i)$$
# MAGIC
# MAGIC **Propiedades:**
# MAGIC - $H(S) = 0$ si todos los ejemplos son de la misma clase
# MAGIC - $0 \\leq H(S) \\leq \\log_2(c)$
# MAGIC
# MAGIC ### 3.2 Ganancia de Información
# MAGIC
# MAGIC $$IG(S, A) = H(S) - \\sum_{v \\in \\text{Valores}(A)} \\frac{|S_v|}{|S|} H(S_v)$$
# MAGIC
# MAGIC **El atributo con mayor ganancia se selecciona para dividir.**

# COMMAND ----------

# DBTITLE 1,Índice Gini
# MAGIC %md
# MAGIC ## 4. Índice Gini
# MAGIC
# MAGIC ### Impureza de Gini
# MAGIC
# MAGIC $$\\text{Gini}(S) = 1 - \\sum_{i=1}^{c} p_i^2$$
# MAGIC
# MAGIC **Ejemplo:** 10 ejemplos: 7 clase A, 3 clase B
# MAGIC
# MAGIC $$\\text{Gini}(S) = 1 - [(0.7)^2 + (0.3)^2] = 1 - 0.58 = 0.42$$
# MAGIC
# MAGIC ### Ganancia Gini
# MAGIC
# MAGIC $$\\text{Gini Gain}(S, A) = \\text{Gini}(S) - \\sum_{v} \\frac{|S_v|}{|S|} \\text{Gini}(S_v)$$
# MAGIC
# MAGIC | Característica | Entropía | Gini |
# MAGIC |----------------|----------|------|
# MAGIC | Cómputo | Logaritmo | Cuadrados |
# MAGIC | Velocidad | Más lento | Más rápido |
# MAGIC | Uso | ID3, C4.5 | CART, Random Forest |

# COMMAND ----------

# DBTITLE 1,Poda y Overfitting
# MAGIC %md
# MAGIC ## 5. Poda (Pruning) y Control de Overfitting
# MAGIC
# MAGIC ### Problema del Overfitting
# MAGIC
# MAGIC Árboles muy profundos memorizan ruido y no generalizan.
# MAGIC
# MAGIC ### 5.1 Poda Pre-procesamiento (Pre-pruning)
# MAGIC
# MAGIC Detener crecimiento antes de tiempo:
# MAGIC
# MAGIC * **maxDepth**: Profundidad máxima del árbol
# MAGIC * **minInstancesPerNode**: Mínimo de ejemplos por nodo
# MAGIC * **minInfoGain**: Ganancia mínima para dividir
# MAGIC
# MAGIC ### 5.2 Poda Post-procesamiento (Post-pruning)
# MAGIC
# MAGIC 1. Crecer árbol completo
# MAGIC 2. Remover subárboles que no mejoran validación
# MAGIC
# MAGIC **Técnicas:**
# MAGIC * **Reduced Error Pruning**: Eliminar subárbol si mejora accuracy en validación
# MAGIC * **Cost Complexity Pruning**: Penalizar árboles grandes
# MAGIC
# MAGIC $$\\text{Costo}_{\\alpha}(T) = \\text{Error}(T) + \\alpha |T|$$
# MAGIC
# MAGIC Donde $|T|$ es el número de hojas y $\\alpha$ controla la penalización.

# COMMAND ----------

# DBTITLE 1,Ventajas y Desventajas
# MAGIC %md
# MAGIC ## 6. Ventajas y Desventajas
# MAGIC
# MAGIC ### ✅ Ventajas
# MAGIC
# MAGIC * **Interpretables**: Fáciles de visualizar y explicar
# MAGIC * **No paramétricos**: Sin supuestos sobre distribución de datos
# MAGIC * **Múltiples tipos de datos**: Numéricos y categóricos
# MAGIC * **Importancia de características**: Identifica variables relevantes
# MAGIC * **Capturan no-linealidades**: Sin transformación manual
# MAGIC
# MAGIC ### ❌ Desventajas
# MAGIC
# MAGIC * **Inestabilidad**: Pequeños cambios en datos producen árboles diferentes
# MAGIC * **Overfitting**: Sin poda, memorizan ruido
# MAGIC * **Sesgos**: Favorecen atributos con muchos valores
# MAGIC * **Fronteras de decisión**: Solo cortes ortogonales (paralelos a ejes)
# MAGIC * **Desbalance**: Problemas con clases desbalanceadas
# MAGIC
# MAGIC ### Soluciones
# MAGIC
# MAGIC * **Random Forests**: Ensemble de árboles reduce inestabilidad
# MAGIC * **Gradient Boosting**: Combina árboles débiles secuencialmente
# MAGIC * **Poda**: Controla complejidad

# COMMAND ----------

# DBTITLE 1,Conclusiones
# MAGIC %md
# MAGIC ## 7. Conclusiones
# MAGIC
# MAGIC ### Conceptos Clave
# MAGIC
# MAGIC 1. **Estructura jerárquica** de decisiones
# MAGIC 2. **Criterios de división**: Entropía (ID3) vs Gini (CART)
# MAGIC 3. **Control de complejidad**: Poda pre y post
# MAGIC 4. **Balance**: Interpretabilidad vs Performance
# MAGIC
# MAGIC ### Próximos Pasos
# MAGIC
# MAGIC * **Random Forests**: Mejoran estabilidad y accuracy
# MAGIC * **Gradient Boosting**: XGBoost, LightGBM
# MAGIC * **Implementación práctica**: Ver notebook práctico

# COMMAND ----------

