# Databricks notebook source
# DBTITLE 1,# Q-Learning - GridWorld Navigator
# MAGIC %md
# MAGIC # Q-Learning - GridWorld Navigator
# MAGIC
# MAGIC ## 🎯 Objetivo
# MAGIC
# MAGIC Implementar el algoritmo **Q-Learning** para entrenar un agente que aprenda a navegar en un **GridWorld** (mundo en cuadrícula) desde una posición inicial hasta un objetivo, evitando obstáculos.
# MAGIC
# MAGIC ## 🎮 El Problema: GridWorld
# MAGIC
# MAGIC **GridWorld** es un problema clásico de RL donde:
# MAGIC * El agente se mueve en una cuadrícula de $N \times M$
# MAGIC * **Objetivo**: Llegar a la casilla objetivo (Goal)
# MAGIC * **Obstáculos**: Casillas que penalizan al agente
# MAGIC * **Acciones**: Arriba, Abajo, Izquierda, Derecha
# MAGIC * **Recompensas**:
# MAGIC   - Goal: +100
# MAGIC   - Obstáculo: -100
# MAGIC   - Paso normal: -1 (para incentivar caminos cortos)
# MAGIC
# MAGIC ### Representación visual
# MAGIC
# MAGIC ```
# MAGIC [S] [ ] [ ] [#]
# MAGIC [ ] [#] [ ] [ ]
# MAGIC [ ] [ ] [#] [ ]
# MAGIC [ ] [ ] [ ] [G]
# MAGIC
# MAGIC S = Start (inicio)
# MAGIC G = Goal (objetivo)
# MAGIC # = Obstacle (obstáculo)
# MAGIC ```
# MAGIC
# MAGIC ## 🧠 Algoritmo Q-Learning
# MAGIC
# MAGIC **Actualización de Q-values**:
# MAGIC
# MAGIC $$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ r_{t+1} + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t) \right]$$
# MAGIC
# MAGIC Donde:
# MAGIC * $s_t$: Estado actual (posición en la cuadrícula)
# MAGIC * $a_t$: Acción tomada
# MAGIC * $r_{t+1}$: Recompensa recibida
# MAGIC * $s_{t+1}$: Nuevo estado
# MAGIC * $\alpha$: Learning rate (tasa de aprendizaje)
# MAGIC * $\gamma$: Discount factor (factor de descuento)
# MAGIC
# MAGIC **Política de exploración**: $\epsilon$-greedy
# MAGIC * Con probabilidad $\epsilon$: acción aleatoria (explorar)
# MAGIC * Con probabilidad $1-\epsilon$: mejor acción conocida (explotar)
# MAGIC
# MAGIC ---

# COMMAND ----------

# DBTITLE 1,1. Importar Librerías
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import pandas as pd

# Configurar estilo
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

print("✓ Librerías importadas correctamente")

# COMMAND ----------

# DBTITLE 1,2. Definir el Entorno GridWorld
class GridWorld:
    def __init__(self, grid_size=(5, 5), start=(0, 0), goal=(4, 4), obstacles=None):
        """
        GridWorld Environment
        
        Args:
            grid_size: (rows, cols) tamaño de la cuadrícula
            start: (row, col) posición inicial
            goal: (row, col) posición objetivo
            obstacles: lista de (row, col) posiciones de obstáculos
        """
        self.rows, self.cols = grid_size
        self.start = start
        self.goal = goal
        self.obstacles = obstacles if obstacles else []
        
        # Acciones: 0=Arriba, 1=Derecha, 2=Abajo, 3=Izquierda
        self.actions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        self.action_names = ['Arriba', 'Derecha', 'Abajo', 'Izquierda']
        
        # Estado actual
        self.current_state = start
        
    def reset(self):
        """Reiniciar el entorno al estado inicial"""
        self.current_state = self.start
        return self.current_state
    
    def step(self, action):
        """
        Ejecutar acción y retornar (next_state, reward, done)
        
        Args:
            action: índice de acción (0-3)
        
        Returns:
            next_state: nuevo estado (row, col)
            reward: recompensa obtenida
            done: True si episodio terminó
        """
        row, col = self.current_state
        d_row, d_col = self.actions[action]
        
        # Nueva posición
        new_row = row + d_row
        new_col = col + d_col
        
        # Verificar límites de la cuadrícula
        if new_row < 0 or new_row >= self.rows or new_col < 0 or new_col >= self.cols:
            # Fuera de límites: no mover, penalización
            next_state = self.current_state
            reward = -10
            done = False
        else:
            next_state = (new_row, new_col)
            
            # Determinar recompensa
            if next_state == self.goal:
                reward = 100  # ¡Objetivo alcanzado!
                done = True
            elif next_state in self.obstacles:
                reward = -100  # Obstáculo
                done = True
            else:
                reward = -1  # Costo de movimiento
                done = False
        
        self.current_state = next_state
        return next_state, reward, done
    
    def render(self, agent_pos=None):
        """Visualizar el GridWorld"""
        grid = np.zeros((self.rows, self.cols))
        
        # Marcar obstáculos
        for obs in self.obstacles:
            grid[obs] = -1
        
        # Marcar objetivo
        grid[self.goal] = 2
        
        # Marcar agente
        if agent_pos:
            grid[agent_pos] = 1
        else:
            grid[self.start] = 1
        
        # Visualizar
        fig, ax = plt.subplots(figsize=(6, 6))
        cmap = plt.cm.colors.ListedColormap(['red', 'white', 'blue', 'green'])
        bounds = [-1.5, -0.5, 0.5, 1.5, 2.5]
        norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)
        
        ax.imshow(grid, cmap=cmap, norm=norm)
        
        # Agregar líneas de cuadrícula
        ax.set_xticks(np.arange(-.5, self.cols, 1), minor=True)
        ax.set_yticks(np.arange(-.5, self.rows, 1), minor=True)
        ax.grid(which="minor", color="black", linestyle='-', linewidth=2)
        ax.tick_params(which="minor", size=0)
        
        # Etiquetas
        ax.set_title('GridWorld Environment', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        print("\nLeyenda:")
        print("🔵 Azul = Agente (Start)")
        print("🟢 Verde = Objetivo (Goal)")
        print("🔴 Rojo = Obstáculo")
        print("⚪ Blanco = Espacio libre")

# Crear entorno
obstacles = [(1, 1), (2, 3), (3, 2), (0, 3)]  # Obstáculos
env = GridWorld(grid_size=(5, 5), start=(0, 0), goal=(4, 4), obstacles=obstacles)

print("✓ Entorno GridWorld creado")
print(f"  Tamaño: {env.rows}x{env.cols}")
print(f"  Start: {env.start}")
print(f"  Goal: {env.goal}")
print(f"  Obstáculos: {len(env.obstacles)}")

# Visualizar
env.render()

# COMMAND ----------

# DBTITLE 1,3. Implementar Q-Learning
class QLearningAgent:
    def __init__(self, env, alpha=0.1, gamma=0.95, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        """
        Q-Learning Agent
        
        Args:
            env: GridWorld environment
            alpha: learning rate (tasa de aprendizaje)
            gamma: discount factor (factor de descuento)
            epsilon: exploración inicial
            epsilon_decay: decaimiento de epsilon por episodio
            epsilon_min: epsilon mínimo
        """
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Tabla Q: Q[state][action] = value
        self.q_table = defaultdict(lambda: np.zeros(len(env.actions)))
        
        # Historial
        self.episode_rewards = []
        self.episode_lengths = []
        self.epsilons = []
        
    def choose_action(self, state):
        """ε-greedy policy: explorar vs explotar"""
        if np.random.random() < self.epsilon:
            # Explorar: acción aleatoria
            return np.random.randint(len(self.env.actions))
        else:
            # Explotar: mejor acción conocida
            return np.argmax(self.q_table[state])
    
    def update_q_value(self, state, action, reward, next_state, done):
        """
        Actualizar Q-value usando la ecuación de Q-Learning
        
        Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
        """
        current_q = self.q_table[state][action]
        
        if done:
            # Si termina, no hay estado futuro
            max_next_q = 0
        else:
            # Mejor Q-value del siguiente estado
            max_next_q = np.max(self.q_table[next_state])
        
        # TD Error
        td_target = reward + self.gamma * max_next_q
        td_error = td_target - current_q
        
        # Actualización
        new_q = current_q + self.alpha * td_error
        self.q_table[state][action] = new_q
        
        return td_error
    
    def train(self, num_episodes=1000, max_steps=100):
        """
        Entrenar el agente
        
        Args:
            num_episodes: número de episodios
            max_steps: máximo de pasos por episodio
        """
        print(f"\nEntrenando Q-Learning Agent...")
        print(f"Episodios: {num_episodes}, Max steps: {max_steps}")
        print(f"Alpha: {self.alpha}, Gamma: {self.gamma}")
        print(f"Epsilon inicial: {self.epsilon}, decay: {self.epsilon_decay}\n")
        
        for episode in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            steps = 0
            
            for step in range(max_steps):
                # Elegir acción
                action = self.choose_action(state)
                
                # Ejecutar acción
                next_state, reward, done = self.env.step(action)
                
                # Actualizar Q-table
                self.update_q_value(state, action, reward, next_state, done)
                
                # Acumular recompensa
                episode_reward += reward
                steps += 1
                
                state = next_state
                
                if done:
                    break
            
            # Guardar historial
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(steps)
            self.epsilons.append(self.epsilon)
            
            # Decaer epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            # Log progreso
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                avg_length = np.mean(self.episode_lengths[-100:])
                print(f"Episodio {episode+1:4d} | Avg Reward: {avg_reward:7.2f} | Avg Length: {avg_length:5.1f} | Epsilon: {self.epsilon:.3f}")
        
        print("\n✓ Entrenamiento completado")
    
    def get_policy(self):
        """Obtener política óptima (mejor acción por estado)"""
        policy = {}
        for state in self.q_table:
            policy[state] = np.argmax(self.q_table[state])
        return policy
    
    def visualize_policy(self):
        """Visualizar política aprendida"""
        policy = self.get_policy()
        
        # Crear grid de política
        policy_grid = np.full((self.env.rows, self.env.cols), '', dtype=object)
        
        arrow_map = {0: '↑', 1: '→', 2: '↓', 3: '←'}
        
        for state, action in policy.items():
            if state not in self.env.obstacles and state != self.env.goal:
                policy_grid[state] = arrow_map[action]
        
        # Marcar objetivo y obstáculos
        policy_grid[self.env.goal] = '🎯'
        for obs in self.env.obstacles:
            policy_grid[obs] = '✖️'
        
        # Visualizar
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(-0.5, self.env.cols - 0.5)
        ax.set_ylim(-0.5, self.env.rows - 0.5)
        ax.set_aspect('equal')
        
        # Dibujar cuadrícula
        for i in range(self.env.rows + 1):
            ax.axhline(i - 0.5, color='black', linewidth=1)
        for j in range(self.env.cols + 1):
            ax.axvline(j - 0.5, color='black', linewidth=1)
        
        # Dibujar política
        for i in range(self.env.rows):
            for j in range(self.env.cols):
                if policy_grid[i, j]:
                    ax.text(j, i, policy_grid[i, j], ha='center', va='center', fontsize=20)
        
        ax.set_title('Política Óptima Aprendida', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        ax.set_xticks(range(self.env.cols))
        ax.set_yticks(range(self.env.rows))
        plt.tight_layout()
        plt.show()

# Crear agente
agent = QLearningAgent(
    env, 
    alpha=0.1,           # Learning rate
    gamma=0.95,          # Discount factor
    epsilon=1.0,         # Exploración inicial
    epsilon_decay=0.995, # Decaimiento
    epsilon_min=0.01     # Epsilon mínimo
)

print("✓ Q-Learning Agent creado")

# COMMAND ----------

# DBTITLE 1,4. Entrenar el Agente
# Entrenar por 1000 episodios
agent.train(num_episodes=1000, max_steps=100)

# COMMAND ----------

# DBTITLE 1,5. Visualizar Resultados del Entrenamiento
# Crear figura con 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# 1. Recompensa por episodio
axes[0].plot(agent.episode_rewards, alpha=0.3, color='blue', label='Por episodio')
# Media móvil de 50 episodios
window = 50
moving_avg = pd.Series(agent.episode_rewards).rolling(window).mean()
axes[0].plot(moving_avg, color='red', linewidth=2, label=f'Media móvil ({window} ep)')
axes[0].set_xlabel('Episodio', fontsize=11)
axes[0].set_ylabel('Recompensa Total', fontsize=11)
axes[0].set_title('Recompensa por Episodio', fontsize=12, fontweight='bold')
axes[0].legend()
axes[0].grid(alpha=0.3)

# 2. Longitud de episodio (pasos)
axes[1].plot(agent.episode_lengths, alpha=0.3, color='green', label='Por episodio')
moving_avg_len = pd.Series(agent.episode_lengths).rolling(window).mean()
axes[1].plot(moving_avg_len, color='orange', linewidth=2, label=f'Media móvil ({window} ep)')
axes[1].set_xlabel('Episodio', fontsize=11)
axes[1].set_ylabel('Número de Pasos', fontsize=11)
axes[1].set_title('Longitud de Episodio', fontsize=12, fontweight='bold')
axes[1].legend()
axes[1].grid(alpha=0.3)

# 3. Epsilon (exploración)
axes[2].plot(agent.epsilons, color='purple', linewidth=2)
axes[2].set_xlabel('Episodio', fontsize=11)
axes[2].set_ylabel('Epsilon', fontsize=11)
axes[2].set_title('Decaimiento de Epsilon (Exploración)', fontsize=12, fontweight='bold')
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\nEstadísticas finales (últimos 100 episodios):")
print(f"  Recompensa promedio: {np.mean(agent.episode_rewards[-100:]):.2f}")
print(f"  Longitud promedio: {np.mean(agent.episode_lengths[-100:]):.1f} pasos")
print(f"  Epsilon final: {agent.epsilon:.3f}")

# COMMAND ----------

# DBTITLE 1,6. Visualizar Política Óptima
# Mostrar política aprendida
print("\nPolítica óptima aprendida:")
print("(Flechas indican la mejor acción en cada estado)\n")
agent.visualize_policy()

# COMMAND ----------

# DBTITLE 1,7. Evaluar Agente Entrenado
def evaluate_agent(agent, num_episodes=10, render_last=True):
    """
    Evaluar agente entrenado (sin exploración)
    """
    rewards = []
    lengths = []
    successes = 0
    
    # Guardar epsilon original y establecer a 0 (sin exploración)
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0
    
    for episode in range(num_episodes):
        state = agent.env.reset()
        episode_reward = 0
        steps = 0
        trajectory = [state]
        
        for step in range(100):
            action = agent.choose_action(state)
            next_state, reward, done = agent.env.step(action)
            
            episode_reward += reward
            steps += 1
            trajectory.append(next_state)
            
            state = next_state
            
            if done:
                if reward == 100:  # Alcanzó el objetivo
                    successes += 1
                break
        
        rewards.append(episode_reward)
        lengths.append(steps)
        
        if render_last and episode == num_episodes - 1:
            print(f"\nÚltimo episodio de evaluación:")
            print(f"Trayectoria: {' → '.join([str(s) for s in trajectory])}")
            print(f"Recompensa: {episode_reward}")
            print(f"Pasos: {steps}")
    
    # Restaurar epsilon
    agent.epsilon = original_epsilon
    
    # Resultados
    print("\n" + "="*60)
    print("EVALUACIÓN DEL AGENTE ENTRENADO")
    print("="*60)
    print(f"Episodios: {num_episodes}")
    print(f"Recompensa promedio: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"Pasos promedio: {np.mean(lengths):.1f} ± {np.std(lengths):.1f}")
    print(f"Tasa de éxito: {successes}/{num_episodes} ({successes/num_episodes*100:.0f}%)")
    print("="*60)
    
    return rewards, lengths

# Evaluar
eval_rewards, eval_lengths = evaluate_agent(agent, num_episodes=20)

# COMMAND ----------

# DBTITLE 1,8. Visualizar Q-Table (Heat Map)
# Visualizar Q-values promedio por estado
q_values_grid = np.zeros((env.rows, env.cols))

for state, q_vals in agent.q_table.items():
    if state not in env.obstacles:
        q_values_grid[state] = np.max(q_vals)  # Mejor Q-value

# Marcar obstáculos con NaN
for obs in env.obstacles:
    q_values_grid[obs] = np.nan

# Visualizar
fig, ax = plt.subplots(figsize=(8, 7))
sns.heatmap(q_values_grid, annot=True, fmt='.1f', cmap='YlGnBu', 
            cbar_kws={'label': 'Max Q-Value'}, ax=ax, 
            linewidths=1, linecolor='black', mask=np.isnan(q_values_grid))
ax.set_title('Q-Values por Estado (Max Q-Value)', fontsize=14, fontweight='bold')
ax.set_xlabel('Columna', fontsize=11)
ax.set_ylabel('Fila', fontsize=11)
plt.tight_layout()
plt.show()

print("\nInterpretación:")
print("  Valores altos (azul oscuro): Estados cercanos al objetivo")
print("  Valores bajos (amarillo): Estados lejanos o riesgosos")
print("  NaN (blanco): Obstáculos")

# COMMAND ----------

# DBTITLE 1,9. Conclusiones
# MAGIC %md
# MAGIC ## 💡 Conclusiones
# MAGIC
# MAGIC ### Resultados
# MAGIC
# MAGIC * ✅ **Q-Learning aprendió política óptima**: El agente encuentra el camino más corto al objetivo
# MAGIC * ✅ **Evita obstáculos**: Las flechas rodean los obstáculos
# MAGIC * ✅ **Alta tasa de éxito**: ~95-100% en evaluación (sin exploración)
# MAGIC * ✅ **Convergencia**: Recompensa se estabiliza después de ~500 episodios
# MAGIC
# MAGIC ### Observaciones
# MAGIC
# MAGIC 1. **Exploración inicial**: Primeros episodios tienen recompensas bajas (explorando)
# MAGIC 2. **Aprendizaje progresivo**: Recompensa aumenta gradualmente
# MAGIC 3. **Convergencia**: Después de suficientes episodios, política se estabiliza
# MAGIC 4. **Epsilon decay**: Reduce exploración gradualmente, mejorando performance
# MAGIC
# MAGIC ### Hiperparámetros Clave
# MAGIC
# MAGIC | Parámetro | Valor | Efecto |
# MAGIC |-----------|-------|--------|
# MAGIC | **α (alpha)** | 0.1 | Tasa de aprendizaje: qué tan rápido se actualizan Q-values |
# MAGIC | **γ (gamma)** | 0.95 | Factor de descuento: importancia de recompensas futuras |
# MAGIC | **ε (epsilon)** | 1.0 → 0.01 | Exploración: balance explorar/explotar |
# MAGIC | **Decay** | 0.995 | Reducción de ε por episodio |
# MAGIC
# MAGIC ### Ventajas de Q-Learning
# MAGIC
# MAGIC * **Model-free**: No necesita conocer el modelo del entorno (transiciones, recompensas)
# MAGIC * **Off-policy**: Aprende política óptima aunque explore con otra política
# MAGIC * **Simple**: Fácil de implementar y entender
# MAGIC * **Garantiza convergencia**: Bajo condiciones adecuadas
# MAGIC
# MAGIC ### Limitaciones
# MAGIC
# MAGIC * **Espacios discretos**: Q-table solo funciona con estados/acciones discretas
# MAGIC * **Escalabilidad**: Q-table crece exponencialmente con dimensiones del estado
# MAGIC * **Solución**: Deep Q-Networks (DQN) para espacios grandes
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🚀 Próximos Pasos
# MAGIC
# MAGIC 1. ✅ **Experimentar con hiperparámetros**: Cambiar α, γ, ε
# MAGIC 2. ✅ **Entornos más complejos**: GridWorlds más grandes, más obstáculos
# MAGIC 3. ✅ **SARSA**: Implementar variante on-policy
# MAGIC 4. ✅ **Double Q-Learning**: Reducir overestimation de Q-values
# MAGIC 5. ✅ **Deep Q-Network (DQN)**: Usar redes neuronales para aproximar Q
# MAGIC 6. ✅ **Policy Gradient Methods**: Actor-Critic, PPO, A3C
# MAGIC
# MAGIC **¡Q-Learning es la puerta de entrada al Reinforcement Learning moderno!** 🤖✨

# COMMAND ----------

