# Aprendizaje por Refuerzo (Reinforcement Learning)

## 🎯 Definición

**Aprendizaje por Refuerzo** (Reinforcement Learning, RL) es un paradigma de Machine Learning donde un **agente** aprende a tomar **decisiones secuenciales** mediante **interacción** con un entorno, recibiendo **recompensas** o penalizaciones por sus acciones, con el objetivo de **maximizar la recompensa acumulada** a largo plazo.

### Características Clave

* 🤖 **Agente autónomo**: Aprende mediante prueba y error
* 🌍 **Entorno dinámico**: Responde a las acciones del agente
* 💰 **Recompensas diferidas**: Consecuencias a largo plazo
* ⚖️ **Exploración vs Explotación**: Balance entre probar y aprovechar
* 📈 **Aprendizaje secuencial**: Decisiones en múltiples pasos

## 📁 Contenido

### 📖 Teoría

**Teoria_Reinforcement_Learning.ipynb** (3 celdas markdown, ~2000 líneas)
- **Conceptos fundamentales**: Agente, entorno, estado, acción, recompensa, política
- **MDP (Markov Decision Process)**: Modelado formal de problemas de RL
- **Funciones de valor**: $V^\pi(s)$ y $Q^\pi(s, a)$
- **Ecuación de Bellman**: Relación recursiva entre valores
- **Exploración vs Explotación**: ε-greedy, Softmax, UCB
- **Métodos de resolución**:
  - Programación Dinámica (Value/Policy Iteration)
  - Monte Carlo
  - Temporal Difference (TD)
- **Algoritmos clásicos**:
  - **Q-Learning**: Off-policy TD control
  - **SARSA**: On-policy TD control
  - **Deep Q-Network (DQN)**: Q-Learning con redes neuronales
- **Aplicaciones**: Juegos, robótica, finanzas, healthcare, vehículos autónomos
- **Algoritmos modernos**: PPO, SAC, TD3, Rainbow, MuZero

**Fórmulas clave:**
- Retorno: $G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k+1}$
- Q-Learning: $Q(s,a) \leftarrow Q(s,a) + \alpha[r + \gamma \max_{a'} Q(s',a') - Q(s,a)]$
- Bellman para $Q^*$: $Q^*(s,a) = R(s,a) + \gamma \sum_{s'} P(s'|s,a) \max_{a'} Q^*(s',a')$

### 💻 Práctica

**Q_Learning_GridWorld.ipynb** (10 celdas)
- **Algoritmo**: Q-Learning (Temporal Difference)
- **Problema**: GridWorld Navigator - Agente aprende a navegar en cuadrícula
- **Entorno**: Grid 5x5 con inicio, objetivo y obstáculos
- **Implementación**: Clase GridWorld y QLearningAgent desde cero
- **Características**:
  - Q-table para almacenar valores
  - ε-greedy exploration policy
  - Visualización de política óptima aprendida
  - Heatmap de Q-values
  - Métricas de entrenamiento (reward, length, epsilon)
- **Resultados**: ~95-100% tasa de éxito tras entrenamiento
- **Hiperparámetros**: α=0.1, γ=0.95, ε: 1.0 → 0.01

## 🎯 Objetivo

Aprender cómo un agente autónomo puede aprender comportamientos complejos mediante **interacción** con un entorno y **optimización** de recompensas acumuladas a largo plazo.

## 📊 Concepto: Reinforcement Learning

RL se diferencia de aprendizaje supervisado y no supervisado:

| Aspecto | Supervisado | No Supervisado | **Por Refuerzo** |
|---------|-------------|----------------|------------------|
| **Datos** | Etiquetados | Sin etiquetas | **Interacción** |
| **Objetivo** | Predecir | Descubrir estructura | **Maximizar recompensa** |
| **Feedback** | Inmediato | Sin feedback | **Diferido, parcial** |
| **Ejemplos** | Clasificación | Clustering | **Juegos, robótica** |

### Componentes clave

```
     ┌─────────────┐
     │   Entorno   │
     │  (Mundo)    │
     └──────┬──────┘
            │ Estado (s_t)
            │ Recompensa (r_t)
            ↓
     ┌─────────────┐
     │   Agente    │
     │  (Decisor)  │
     └──────┬──────┘
            │ Acción (a_t)
            ↓
     [Ciclo se repite]
```

1. **Estado** ($s_t$): Situación actual del entorno
2. **Acción** ($a_t$): Decisión que toma el agente
3. **Recompensa** ($r_t$): Señal numérica del entorno
4. **Política** ($\pi$): Estrategia que mapea estados a acciones
5. **Función de valor**: Estima "bondad" de un estado o acción

## 🚀 Orden de Estudio Recomendado

### Para Principiantes:

1. **Fundamentos** (carpeta `../Fundamentos/`)
   - Introducción a Machine Learning
   - Conceptos matemáticos básicos

2. **Teoría** (`Teoria_Reinforcement_Learning`)
   - Leer secuencialmente todas las secciones
   - Comprender MDP, funciones de valor, Bellman
   - Estudiar Q-Learning (el algoritmo más importante)
   - Entender exploración vs explotación

3. **Práctica** (`Q_Learning_GridWorld`)
   - Ejecutar todas las celdas secuencialmente
   - Observar cómo el agente aprende la política óptima
   - Experimentar con hiperparámetros (α, γ, ε)
   - Cambiar el GridWorld (más grande, más obstáculos)

### Para Avanzados:

1. Implementar SARSA (on-policy)
2. Explorar Deep Q-Network (DQN) con redes neuronales
3. Policy Gradient Methods (REINFORCE, Actor-Critic)
4. Algoritmos modernos (PPO, SAC, TD3)
5. Ambientes complejos (OpenAI Gym, Unity ML-Agents)

## 💡 Cómo usar estos notebooks

### Notebook Teórico:
1. Lee secuencialmente todas las celdas
2. Toma notas de las ecuaciones clave (Bellman, Q-Learning)
3. Comprende la diferencia entre on-policy y off-policy
4. No ejecutes (es contenido markdown)

### Notebook Práctico:
1. Asegúrate de tener un cluster activo (o serverless)
2. Ejecuta todas las celdas en orden
3. Observa:
   - Convergencia de recompensas
   - Decaimiento de epsilon
   - Política óptima aprendida (flechas)
   - Heatmap de Q-values
4. Experimenta:
   - Cambiar α (learning rate): valores más altos → aprendizaje más rápido pero inestable
   - Cambiar γ (discount factor): valores más altos → agente más "visionario"
   - Cambiar ε decay: más lento → más exploración
   - Modificar el GridWorld: agregar obstáculos, cambiar posiciones

## 📚 Conceptos Clave

### Dilema Exploración-Explotación

* **Exploración**: Probar acciones nuevas para descubrir mejores estrategias
* **Explotación**: Elegir la mejor acción conocida hasta ahora
* **Solución**: ε-greedy, Softmax, UCB

### Diferencia Q-Learning vs SARSA

| Aspecto | Q-Learning | SARSA |
|---------|------------|-------|
| **Tipo** | Off-policy | On-policy |
| **Update** | $\max_{a'} Q(s',a')$ | $Q(s', a')$ (acción real) |
| **Comportamiento** | Agresivo, aprende óptimo directamente | Conservador, aprende política actual |
| **Uso** | Cuando entorno es determinista | Cuando entorno es estocástico |

### Convergencia

**Q-Learning converge a Q*** si:
1. Todos los pares (estado, acción) se visitan infinitas veces
2. Learning rate α decrece adecuadamente: $\sum \alpha_t = \infty$ y $\sum \alpha_t^2 < \infty$

En práctica: usar α constante pequeño (0.01-0.1) o decay suave

## 💼 Aplicaciones Reales

### 🎮 Juegos
* **AlphaGo** (DeepMind): Derrotó al campeón mundial de Go
* **OpenAI Five**: Dota 2 profesional
* **AlphaStar**: StarCraft II
* **Atari Games**: DQN jugando 49 juegos

### 🤖 Robótica
* Manipulación: Agarre de objetos, ensamblaje
* Locomoción: Robots bípedos, cuadrúpedos
* Navegación: Drones, vehículos autónomos

### 💰 Finanzas
* Trading algorítmico: Compra/venta de acciones
* Portfolio optimization: Asignación óptima de activos
* Market making: Provisión de liquidez

### 🏥 Healthcare
* Tratamientos personalizados: Dosis óptimas de medicamentos
* Radiología: Planificación de radioterapia

### 🚗 Vehículos Autónomos
* Control: Aceleración, frenado, dirección
* Planificación de rutas: Optimización de trayectorias

### ⚡ Optimización de Recursos
* **Data centers**: Enfriamiento eficiente (Google redujo 40% costos)
* Energía: Gestión de redes eléctricas
* Manufactura: Scheduling de producción

## 🛠️ Herramientas y Frameworks

* **OpenAI Gym**: Entornos estándar de RL (CartPole, MountainCar, Atari)
* **Stable Baselines3**: Implementaciones de algoritmos SOTA (PPO, SAC, DQN)
* **Ray RLlib**: RL distribuido y escalable
* **TF-Agents / PyTorch RL**: Bibliotecas de deep RL
* **Unity ML-Agents**: RL en entornos Unity 3D
* **PettingZoo**: Multi-agent RL

## 📚 Recursos

### Libros
* **"Reinforcement Learning: An Introduction"** - Sutton & Barto (biblia de RL, gratis online)
* **"Deep Reinforcement Learning Hands-On"** - Maxim Lapan
* **"Algorithms for Reinforcement Learning"** - Csaba Szepesvári

### Cursos
* **CS 285** - Deep RL (UC Berkeley) - [Gratis online](http://rail.eecs.berkeley.edu/deeprlcourse/)
* **David Silver's RL Course** (DeepMind) - [YouTube](https://www.youtube.com/playlist?list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ)
* **Coursera**: Reinforcement Learning Specialization (University of Alberta)

### Papers clave
* **DQN**: Mnih et al., 2015 - "Human-level control through deep RL"
* **AlphaGo**: Silver et al., 2016 - "Mastering the game of Go"
* **PPO**: Schulman et al., 2017 - "Proximal Policy Optimization"
* **SAC**: Haarnoja et al., 2018 - "Soft Actor-Critic"

### Comunidades
* [Reddit r/reinforcementlearning](https://www.reddit.com/r/reinforcementlearning/)
* [OpenAI Spinning Up](https://spinningup.openai.com/) - Tutorial educativo de OpenAI
* [Hugging Face RL Course](https://huggingface.co/deep-rl-course) - Curso interactivo gratis

---

**Siguiente paso**: Experimenta con el notebook práctico y observa cómo el agente aprende a navegar el GridWorld. Luego, explora entornos más complejos como OpenAI Gym.

**¡El aprendizaje por refuerzo está revolucionando la IA, permitiendo agentes autónomos que aprenden por sí mismos!** 🤖✨
