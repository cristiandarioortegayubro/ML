# Databricks notebook source
# DBTITLE 1,# Aprendizaje por Refuerzo - Teoría
# MAGIC %md
# MAGIC # Aprendizaje por Refuerzo (Reinforcement Learning)
# MAGIC
# MAGIC ## 🎯 Definición
# MAGIC
# MAGIC **Aprendizaje por Refuerzo** (Reinforcement Learning, RL) es un paradigma de Machine Learning donde un **agente** aprende a tomar **decisiones** interactuando con un **entorno**, recibiendo **recompensas** (o penalizaciones) por sus acciones, con el objetivo de **maximizar la recompensa acumulada** a largo plazo.
# MAGIC
# MAGIC ### Diferencia con otros paradigmas
# MAGIC
# MAGIC | Paradigma | Datos | Objetivo | Feedback |
# MAGIC |-----------|-------|----------|----------|
# MAGIC | **Supervisado** | Etiquetados | Predecir etiqueta correcta | Inmediato (correcto/incorrecto) |
# MAGIC | **No Supervisado** | Sin etiquetas | Descubrir estructura | Sin feedback explícito |
# MAGIC | **Por Refuerzo** | Interacción | Maximizar recompensa acumulada | Diferido, parcial, escaso |
# MAGIC
# MAGIC ### Características clave
# MAGIC
# MAGIC * 🤖 **Agente autónomo**: Toma decisiones secuenciales
# MAGIC * 🌍 **Entorno dinámico**: Responde a las acciones del agente
# MAGIC * 💰 **Recompensas diferidas**: El resultado de una acción puede verse mucho después
# MAGIC * ⚖️ **Trade-off exploración/explotación**: Balance entre explorar opciones nuevas y explotar conocimiento actual
# MAGIC * 📈 **Aprendizaje por prueba y error**: El agente mejora mediante experiencia
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 📘 Conceptos Fundamentales
# MAGIC
# MAGIC ### Componentes de un sistema de RL
# MAGIC
# MAGIC ```
# MAGIC      ┌─────────────┐
# MAGIC      │   Entorno   │
# MAGIC      │  (Mundo)    │
# MAGIC      └──────┬──────┘
# MAGIC             │ Estado (s_t)
# MAGIC             │ Recompensa (r_t)
# MAGIC             ↓
# MAGIC      ┌─────────────┐
# MAGIC      │   Agente    │
# MAGIC      │  (Decisor)  │
# MAGIC      └──────┬──────┘
# MAGIC             │ Acción (a_t)
# MAGIC             ↓
# MAGIC      [Ciclo se repite]
# MAGIC ```
# MAGIC
# MAGIC 1. **Agente**: Entidad que toma decisiones (ej: robot, jugador de ajedrez, sistema de trading)
# MAGIC 2. **Entorno**: Mundo con el que interactúa el agente (ej: tablero, mercado financiero)
# MAGIC 3. **Estado** ($s_t$): Representación de la situación actual del entorno en el tiempo $t$
# MAGIC 4. **Acción** ($a_t$): Decisión que toma el agente en el estado $s_t$
# MAGIC 5. **Recompensa** ($r_t$): Señal numérica que el entorno da al agente tras una acción
# MAGIC 6. **Política** ($\pi$): Estrategia que mapea estados a acciones: $\pi(a|s)$
# MAGIC 7. **Función de valor**: Estima la "bondad" de un estado o par (estado, acción)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🔄 Proceso Markoviano de Decisión (MDP)
# MAGIC
# MAGIC RL se modela típicamente como un **MDP (Markov Decision Process)**:
# MAGIC
# MAGIC $$\text{MDP} = \langle \mathcal{S}, \mathcal{A}, P, R, \gamma \rangle$$
# MAGIC
# MAGIC Donde:
# MAGIC * $\mathcal{S}$: Conjunto de estados
# MAGIC * $\mathcal{A}$: Conjunto de acciones
# MAGIC * $P(s'|s, a)$: Probabilidad de transición al estado $s'$ dado estado $s$ y acción $a$
# MAGIC * $R(s, a)$: Recompensa esperada al tomar acción $a$ en estado $s$
# MAGIC * $\gamma \in [0, 1]$: Factor de descuento (importancia de recompensas futuras)
# MAGIC
# MAGIC ### Propiedad de Markov
# MAGIC
# MAGIC El futuro depende **solo del estado presente**, no del pasado:
# MAGIC
# MAGIC $$P(s_{t+1} | s_t, a_t, s_{t-1}, a_{t-1}, ..., s_0, a_0) = P(s_{t+1} | s_t, a_t)$$
# MAGIC
# MAGIC Esto simplifica el problema: el estado $s_t$ contiene toda la información relevante.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 💡 Objetivo del Agente
# MAGIC
# MAGIC El objetivo es encontrar una **política óptima** $\pi^*$ que maximice el **retorno esperado** $G_t$ (recompensa acumulada descontada):
# MAGIC
# MAGIC $$G_t = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + ... = \sum_{k=0}^{\infty} \gamma^k r_{t+k+1}$$
# MAGIC
# MAGIC Donde:
# MAGIC * $r_{t+1}$: Recompensa inmediata en $t+1$
# MAGIC * $\gamma$: Factor de descuento ($0 \leq \gamma < 1$)
# MAGIC   - $\gamma = 0$: Solo importa recompensa inmediata (miope)
# MAGIC   - $\gamma \to 1$: Todas las recompensas futuras importan igual (visionario)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 📊 Funciones de Valor
# MAGIC
# MAGIC ### 1. Función de Valor de Estado: $V^\pi(s)$
# MAGIC
# MAGIC **Valor** de estar en estado $s$ siguiendo política $\pi$:
# MAGIC
# MAGIC $$V^\pi(s) = \mathbb{E}_\pi[G_t | s_t = s] = \mathbb{E}_\pi \left[ \sum_{k=0}^{\infty} \gamma^k r_{t+k+1} \mid s_t = s \right]$$
# MAGIC
# MAGIC **Interpretación**: "¿Qué tan bueno es estar en este estado si sigo mi política $\pi$?"
# MAGIC
# MAGIC ### 2. Función de Valor de Acción (Q-Function): $Q^\pi(s, a)$
# MAGIC
# MAGIC **Valor** de tomar acción $a$ en estado $s$ y luego seguir política $\pi$:
# MAGIC
# MAGIC $$Q^\pi(s, a) = \mathbb{E}_\pi[G_t | s_t = s, a_t = a]$$
# MAGIC
# MAGIC **Interpretación**: "¿Qué tan bueno es tomar esta acción en este estado?"
# MAGIC
# MAGIC ### Relación entre $V$ y $Q$
# MAGIC
# MAGIC $$V^\pi(s) = \sum_{a \in \mathcal{A}} \pi(a|s) Q^\pi(s, a)$$
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🏆 Política Óptima
# MAGIC
# MAGIC ### Función de Valor Óptima
# MAGIC
# MAGIC $$V^*(s) = \max_\pi V^\pi(s) \quad \text{(mejor valor posible en estado } s \text{)}$$
# MAGIC
# MAGIC $$Q^*(s, a) = \max_\pi Q^\pi(s, a) \quad \text{(mejor valor de tomar acción } a \text{ en } s \text{)}$$
# MAGIC
# MAGIC ### Política Óptima $\pi^*$
# MAGIC
# MAGIC Una política es **óptima** si $V^{\pi^*}(s) \geq V^\pi(s)$ para todo $s$ y toda política $\pi$.
# MAGIC
# MAGIC Si conocemos $Q^*(s, a)$, la política óptima es:
# MAGIC
# MAGIC $$\pi^*(s) = \arg\max_a Q^*(s, a)$$
# MAGIC
# MAGIC **Ecuación de Bellman para $Q^*$**:
# MAGIC
# MAGIC $$Q^*(s, a) = R(s, a) + \gamma \sum_{s'} P(s'|s, a) \max_{a'} Q^*(s', a')$$
# MAGIC
# MAGIC Esta ecuación recursiva es la base de muchos algoritmos de RL.

# COMMAND ----------

# DBTITLE 1,## 🎲 Exploración vs Explotación
# MAGIC %md
# MAGIC ## 🎲 Exploración vs Explotación
# MAGIC
# MAGIC Uno de los dilemas centrales en RL:
# MAGIC
# MAGIC * **Explotación** (Exploitation): Elegir la mejor acción conocida hasta ahora
# MAGIC   - Maximiza recompensa inmediata
# MAGIC   - Puede quedar atrapado en óptimos locales
# MAGIC
# MAGIC * **Exploración** (Exploration): Probar acciones nuevas
# MAGIC   - Puede descubrir mejores estrategias
# MAGIC   - Sacrifica recompensa a corto plazo
# MAGIC
# MAGIC ### Estrategias comunes
# MAGIC
# MAGIC #### 1. ε-Greedy (Epsilon-Greedy)
# MAGIC
# MAGIC Con probabilidad $\epsilon$: elegir acción aleatoria (explorar)  
# MAGIC Con probabilidad $1 - \epsilon$: elegir mejor acción conocida (explotar)
# MAGIC
# MAGIC $$
# MAGIC \pi(a|s) = 
# MAGIC \begin{cases}
# MAGIC 1 - \epsilon + \frac{\epsilon}{|\mathcal{A}|} & \text{si } a = \arg\max_{a'} Q(s, a') \\
# MAGIC \frac{\epsilon}{|\mathcal{A}|} & \text{en otro caso}
# MAGIC \end{cases}
# MAGIC $$
# MAGIC
# MAGIC **Decay de $\epsilon$**: Empezar con $\epsilon$ alto (explorar mucho) y reducirlo gradualmente:
# MAGIC
# MAGIC $$\epsilon_t = \epsilon_{\text{min}} + (\epsilon_{\text{max}} - \epsilon_{\text{min}}) e^{-\lambda t}$$
# MAGIC
# MAGIC #### 2. Softmax / Boltzmann Exploration
# MAGIC
# MAGIC Elegir acciones con probabilidad proporcional a su Q-value:
# MAGIC
# MAGIC $$\pi(a|s) = \frac{e^{Q(s,a) / \tau}}{\sum_{a'} e^{Q(s,a') / \tau}}$$
# MAGIC
# MAGIC Donde $\tau$ (temperatura) controla cuán "aleatorio" es:
# MAGIC * $\tau \to 0$: Casi determinista (siempre mejor acción)
# MAGIC * $\tau \to \infty$: Uniforme (completamente aleatorio)
# MAGIC
# MAGIC #### 3. Upper Confidence Bound (UCB)
# MAGIC
# MAGIC Elegir la acción que maximiza:
# MAGIC
# MAGIC $$a_t = \arg\max_a \left[ Q(s, a) + c \sqrt{\frac{\ln t}{N(s, a)}} \right]$$
# MAGIC
# MAGIC Donde:
# MAGIC * $N(s, a)$: Número de veces que se ha elegido acción $a$ en estado $s$
# MAGIC * $c$: Constante que controla exploración
# MAGIC * El término $\sqrt{\ln t / N(s,a)}$ favorece acciones poco probadas
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🔧 Métodos de Resolución
# MAGIC
# MAGIC Existen tres familias principales de algoritmos:
# MAGIC
# MAGIC ### 1️⃣ Programación Dinámica (DP)
# MAGIC
# MAGIC **Requisito**: Conocer completamente el modelo del MDP ($P$, $R$)
# MAGIC
# MAGIC #### Value Iteration
# MAGIC
# MAGIC Actualizar iterativamente:
# MAGIC
# MAGIC $$V_{k+1}(s) = \max_a \left[ R(s, a) + \gamma \sum_{s'} P(s'|s,a) V_k(s') \right]$$
# MAGIC
# MAGIC Hasta convergencia: $V_k \to V^*$
# MAGIC
# MAGIC #### Policy Iteration
# MAGIC
# MAGIC 1. **Policy Evaluation**: Calcular $V^\pi$ para política actual
# MAGIC 2. **Policy Improvement**: Mejorar política: $\pi'(s) = \arg\max_a Q^\pi(s, a)$
# MAGIC 3. Repetir hasta que $\pi$ no cambie
# MAGIC
# MAGIC **Ventaja**: Convergencia garantizada  
# MAGIC **Desventaja**: Requiere modelo completo (raro en práctica)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 2️⃣ Métodos Monte Carlo (MC)
# MAGIC
# MAGIC **Idea**: Aprender de **episodios completos** de experiencia
# MAGIC
# MAGIC **Algoritmo básico** (First-Visit MC):
# MAGIC
# MAGIC 1. Generar un episodio siguiendo política $\pi$: $s_0, a_0, r_1, s_1, a_1, r_2, ..., s_T$
# MAGIC 2. Para cada estado $s$ visitado:
# MAGIC    - Calcular retorno $G_t = r_{t+1} + \gamma r_{t+2} + ... + \gamma^{T-t-1} r_T$
# MAGIC    - Actualizar: $V(s) \leftarrow V(s) + \alpha [G_t - V(s)]$
# MAGIC
# MAGIC Donde $\alpha$ es la tasa de aprendizaje.
# MAGIC
# MAGIC **Ventaja**: No necesita modelo del entorno  
# MAGIC **Desventaja**: Solo aprende al final del episodio (puede ser lento)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 3️⃣ Métodos de Diferencia Temporal (TD)
# MAGIC
# MAGIC **Idea**: Combinar MC y DP: aprender en **cada paso** usando estimaciones
# MAGIC
# MAGIC #### TD(0) - Temporal Difference Learning
# MAGIC
# MAGIC Actualización después de cada paso:
# MAGIC
# MAGIC $$V(s_t) \leftarrow V(s_t) + \alpha [r_{t+1} + \gamma V(s_{t+1}) - V(s_t)]$$
# MAGIC
# MAGIC Donde:
# MAGIC * $r_{t+1} + \gamma V(s_{t+1})$: **TD Target** (estimación del retorno)
# MAGIC * $\delta_t = r_{t+1} + \gamma V(s_{t+1}) - V(s_t)$: **TD Error**
# MAGIC
# MAGIC **Ventaja**: Aprende en línea, no necesita episodios completos  
# MAGIC **Desventaja**: Usa estimaciones (puede ser inestable)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🧠 Algoritmos Clásicos de RL
# MAGIC
# MAGIC ### 1. Q-Learning (Off-Policy TD)
# MAGIC
# MAGIC **El algoritmo más popular de RL clásico**.
# MAGIC
# MAGIC **Actualización**:
# MAGIC
# MAGIC $$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ r_{t+1} + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t) \right]$$
# MAGIC
# MAGIC **Pseudocódigo**:
# MAGIC
# MAGIC ```
# MAGIC Inicializar Q(s, a) arbitrariamente
# MAGIC Para cada episodio:
# MAGIC     Inicializar s
# MAGIC     Para cada paso del episodio:
# MAGIC         Elegir acción a usando política derivada de Q (ej: ε-greedy)
# MAGIC         Ejecutar a, observar r, s'
# MAGIC         Q(s, a) ← Q(s, a) + α[r + γ max_a' Q(s', a') - Q(s, a)]
# MAGIC         s ← s'
# MAGIC     Fin Para
# MAGIC Fin Para
# MAGIC ```
# MAGIC
# MAGIC **Características**:
# MAGIC * **Off-policy**: Aprende política óptima aunque use otra para explorar
# MAGIC * **Model-free**: No necesita modelo del entorno
# MAGIC * **Converge a Q^***: Bajo ciertas condiciones (visitar todos (s, a) infinitas veces, $\alpha$ decreciente)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 2. SARSA (On-Policy TD)
# MAGIC
# MAGIC **Similar a Q-Learning pero on-policy**.
# MAGIC
# MAGIC **Actualización**:
# MAGIC
# MAGIC $$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ r_{t+1} + \gamma Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t) \right]$$
# MAGIC
# MAGIC Donde $a_{t+1}$ es la acción **realmente tomada** (no $\max$).
# MAGIC
# MAGIC **Diferencia con Q-Learning**:
# MAGIC * Q-Learning: Usa $\max_{a'} Q(s', a')$ (política óptima)
# MAGIC * SARSA: Usa $Q(s', a')$ donde $a'$ fue realmente elegida (política actual)
# MAGIC
# MAGIC **Trade-off**:
# MAGIC * SARSA: Más conservador, aprende la política que realmente sigue
# MAGIC * Q-Learning: Más agresivo, aprende política óptima directamente
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 3. Deep Q-Network (DQN)
# MAGIC
# MAGIC **Idea**: Aproximar $Q(s, a)$ con una **red neuronal** en vez de tabla.
# MAGIC
# MAGIC **Arquitectura**:
# MAGIC
# MAGIC ```
# MAGIC Estado (s) → [Red Neuronal] → Q(s, a₁), Q(s, a₂), ..., Q(s, aₙ)
# MAGIC ```
# MAGIC
# MAGIC **Innovaciones clave** (Mnih et al., 2015):
# MAGIC
# MAGIC 1. **Experience Replay**: Guardar transiciones $(s, a, r, s')$ en buffer, entrenar con minibatches aleatorios
# MAGIC    - Rompe correlaciones entre muestras consecutivas
# MAGIC    - Reutiliza experiencia
# MAGIC
# MAGIC 2. **Target Network**: Red separada $Q_{\text{target}}$ que se actualiza periódicamente
# MAGIC    - Estabiliza el entrenamiento
# MAGIC    - Evita "perseguir un objetivo móvil"
# MAGIC
# MAGIC **Loss function**:
# MAGIC
# MAGIC $$L(\theta) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}} \left[ \left( r + \gamma \max_{a'} Q_{\text{target}}(s', a'; \theta^-) - Q(s, a; \theta) \right)^2 \right]$$
# MAGIC
# MAGIC Donde:
# MAGIC * $\theta$: Parámetros de la red principal
# MAGIC * $\theta^-$: Parámetros de la target network (actualizados cada $C$ pasos)
# MAGIC * $\mathcal{D}$: Replay buffer
# MAGIC
# MAGIC **Aplicaciones famosas**: Atari games, robótica, juegos de estrategia

# COMMAND ----------

# DBTITLE 1,## 🎯 Aplicaciones de Reinforcement Learning
# MAGIC %md
# MAGIC ## 🎯 Aplicaciones de Reinforcement Learning
# MAGIC
# MAGIC ### 🎮 Juegos
# MAGIC
# MAGIC * **AlphaGo** (DeepMind): Derrotó al campeón mundial de Go
# MAGIC * **OpenAI Five**: Dota 2 profesional
# MAGIC * **AlphaStar** (DeepMind): StarCraft II
# MAGIC * **Atari Games**: DQN jugando 49 juegos
# MAGIC
# MAGIC ### 🤖 Robótica
# MAGIC
# MAGIC * **Manipulación**: Agarre de objetos, ensamblaje
# MAGIC * **Locomoción**: Robots bípedos, cuadrúpedos
# MAGIC * **Navegación**: Drones, vehículos autónomos
# MAGIC * **Simulación → Real**: Transfer learning de simulación a mundo real
# MAGIC
# MAGIC ### 💰 Finanzas
# MAGIC
# MAGIC * **Trading algorítmico**: Compra/venta de acciones
# MAGIC * **Portfolio optimization**: Asignación óptima de activos
# MAGIC * **Market making**: Provisión de liquidez
# MAGIC * **Risk management**: Cobertura de riesgos
# MAGIC
# MAGIC ### 🏥 Healthcare
# MAGIC
# MAGIC * **Tratamientos personalizados**: Dosis óptimas de medicamentos
# MAGIC * **Radiología**: Planificación de radioterapia
# MAGIC * **Manejo de recursos**: Asignación de camas, personal
# MAGIC
# MAGIC ### 🚗 Vehículos Autónomos
# MAGIC
# MAGIC * **Control**: Aceleración, frenado, dirección
# MAGIC * **Planificación de rutas**: Optimización de trayectorias
# MAGIC * **Interacción**: Con otros vehículos y peatones
# MAGIC
# MAGIC ### 📺 Recomendaciones
# MAGIC
# MAGIC * **Contenido**: Netflix, YouTube (qué mostrar)
# MAGIC * **Publicidad**: Qué anuncios mostrar y cuándo
# MAGIC * **E-commerce**: Productos a recomendar
# MAGIC
# MAGIC ### ⚡ Optimización de Recursos
# MAGIC
# MAGIC * **Data centers**: Enfriamiento eficiente (Google)
# MAGIC * **Energía**: Gestión de redes eléctricas
# MAGIC * **Manufactura**: Scheduling de producción
# MAGIC * **Logística**: Ruteo de vehículos
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## ⚖️ Ventajas y Desventajas
# MAGIC
# MAGIC ### ✅ Ventajas
# MAGIC
# MAGIC 1. **Autonomía**: Aprende sin supervisión directa
# MAGIC 2. **Adaptabilidad**: Puede adaptarse a entornos cambiantes
# MAGIC 3. **Decisiones secuenciales**: Maneja problemas de planificación a largo plazo
# MAGIC 4. **Descubrimiento**: Puede encontrar estrategias no obvias
# MAGIC
# MAGIC ### ❌ Desventajas
# MAGIC
# MAGIC 1. **Sample efficiency**: Requiere muchas interacciones con el entorno
# MAGIC 2. **Diseño de recompensas**: Difícil definir recompensas correctas
# MAGIC    - **Reward hacking**: Agente explota bugs en recompensas
# MAGIC 3. **Estabilidad**: Algoritmos pueden ser inestables (especialmente con redes neuronales)
# MAGIC 4. **Exploración**: Puede ser difícil explorar entornos grandes
# MAGIC 5. **Costo computacional**: Entrenamiento puede ser muy caro
# MAGIC 6. **Simulación vs realidad**: Gap entre simulación y mundo real
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 📊 Comparación de Algoritmos
# MAGIC
# MAGIC | Algoritmo | Model-Free | On/Off-Policy | Continuous | Pros | Contras |
# MAGIC |-----------|------------|---------------|------------|------|--------|
# MAGIC | **Q-Learning** | ✅ | Off | ❌ | Simple, converge | Solo discreto |
# MAGIC | **SARSA** | ✅ | On | ❌ | Conservador | Solo discreto |
# MAGIC | **DQN** | ✅ | Off | ❌ | Deep, estable | Solo discreto |
# MAGIC | **Policy Gradient** | ✅ | On | ✅ | Continuo | Alta varianza |
# MAGIC | **Actor-Critic** | ✅ | On/Off | ✅ | Balance | Más complejo |
# MAGIC | **PPO** | ✅ | On | ✅ | Estable, popular | Sample efficiency |
# MAGIC | **SAC** | ✅ | Off | ✅ | Sample efficient | Más complejo |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🔮 Futuro y Tendencias
# MAGIC
# MAGIC ### 🚀 Áreas activas de investigación
# MAGIC
# MAGIC 1. **Multi-agent RL**: Múltiples agentes interactuando
# MAGIC 2. **Offline RL**: Aprender de datasets fijos (sin interacción)
# MAGIC 3. **Meta-RL**: Aprender a aprender (adaptación rápida)
# MAGIC 4. **Safe RL**: Garantías de seguridad durante entrenamiento
# MAGIC 5. **Hierarchical RL**: Decisiones a múltiples niveles temporales
# MAGIC 6. **Model-based RL**: Aprender modelo del entorno para planificar
# MAGIC 7. **RL con Large Language Models**: Combinar LLMs con RL
# MAGIC
# MAGIC ### 🌟 Algoritmos modernos
# MAGIC
# MAGIC * **PPO (Proximal Policy Optimization)**: Estable y eficiente
# MAGIC * **SAC (Soft Actor-Critic)**: Maximum entropy RL
# MAGIC * **TD3 (Twin Delayed DDPG)**: Mejora sobre DDPG
# MAGIC * **Rainbow**: Combinación de mejoras a DQN
# MAGIC * **MuZero**: Aprende modelo implícito del mundo
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 📚 Recursos y Referencias
# MAGIC
# MAGIC ### Libros
# MAGIC
# MAGIC * **"Reinforcement Learning: An Introduction"** - Sutton & Barto (biblia de RL)
# MAGIC * **"Deep Reinforcement Learning Hands-On"** - Maxim Lapan
# MAGIC * **"Algorithms for Reinforcement Learning"** - Csaba Szepesvári
# MAGIC
# MAGIC ### Cursos
# MAGIC
# MAGIC * **CS 285** - Deep RL (UC Berkeley)
# MAGIC * **David Silver's RL Course** (DeepMind)
# MAGIC * **Coursera**: Reinforcement Learning Specialization
# MAGIC
# MAGIC ### Papers clave
# MAGIC
# MAGIC * **DQN**: Mnih et al., 2015 - "Human-level control through deep RL"
# MAGIC * **AlphaGo**: Silver et al., 2016 - "Mastering the game of Go with deep neural networks"
# MAGIC * **PPO**: Schulman et al., 2017 - "Proximal Policy Optimization Algorithms"
# MAGIC * **SAC**: Haarnoja et al., 2018 - "Soft Actor-Critic"
# MAGIC
# MAGIC ### Bibliotecas y Frameworks
# MAGIC
# MAGIC * **OpenAI Gym**: Entornos estándar de RL
# MAGIC * **Stable Baselines3**: Implementaciones de algoritmos de RL
# MAGIC * **Ray RLlib**: RL distribuido y escalable
# MAGIC * **TF-Agents / PyTorch RL**: Bibliotecas de deep RL
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 💡 Conclusión
# MAGIC
# MAGIC **Reinforcement Learning** es un paradigma poderoso para problemas de **toma de decisiones secuenciales**. Aunque más complejo que aprendizaje supervisado o no supervisado, permite a los agentes aprender comportamientos sofisticados mediante interacción y experiencia.
# MAGIC
# MAGIC ### Cuándo usar RL
# MAGIC
# MAGIC ✅ **Sí**:
# MAGIC * Tienes un entorno donde puedes simular acciones
# MAGIC * Decisiones secuenciales con consecuencias a largo plazo
# MAGIC * Puedes definir recompensas claras
# MAGIC * Exploración es factible
# MAGIC
# MAGIC ❌ **No**:
# MAGIC * Datos etiquetados abundantes (usa supervisado)
# MAGIC * Una sola decisión (usa clasificación/regresión)
# MAGIC * Recompensas imposibles de definir
# MAGIC * Exploración muy costosa o peligrosa
# MAGIC
# MAGIC **¡El aprendizaje por refuerzo está revolucionando la IA, permitiendo agentes autónomos que aprenden por sí mismos!** 🤖✨

# COMMAND ----------

