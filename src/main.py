from maze import Maze, Test_maze, Test_maze_little
from agent_epsilon_greedy import Agent_epsilon_greedy
import config
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider


env = Test_maze()
env.draw()
agent = Agent_epsilon_greedy(height = env.height,
                             width= env.width,
                             epsilon=0.3, 
                             alpha=0.1, 
                             gamma=0.9)

nb_episodes = 1000
scores = []
epsilons = []
parcours = []

print("Début de l'entraînement...")
for episode in tqdm(range(nb_episodes)):
    env.reset()
    parcour = []
    
    done = False
    state = env.agent_pos
    parcour.append(state)
    while not done:
        action = agent.select_action(state)
        next_state, reward, done = env.step(action)
        while next_state == None:
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
        
        agent.update(state, action, reward, next_state, done, method='q_learning')
        state = next_state
        parcour.append(state)
    parcours.append(parcour)
        
    # Enregistrement du score final
    final_score = env.step_numbers
    scores.append(final_score)
    epsilons.append(agent.epsilon)
    
    # Décroissance d'epsilon
    agent.decay_epsilon()
print(f"Entraînement terminé. Score moyen des 50 derniers épisodes: {np.mean(scores[-50:]):.2f}")

# Affichage des scores
plt.figure(figsize=(12, 5))
plt.plot(scores)
plt.title('Évolution des scores par épisode')
plt.xlabel('Épisode')
plt.ylabel('Nombre de pas')
plt.show()

# Fonction pour dessiner le labyrinthe avec le parcours
def draw_maze_with_path(ax, env, path, episode_num):
    ax.clear()
    
    # Dessiner la grille de base
    for i in range(env.height + 1):
        ax.axhline(y=i, color='lightgray', linewidth=0.5)
    for j in range(env.width + 1):
        ax.axvline(x=j, color='lightgray', linewidth=0.5)
    
    # Dessiner les murs verticaux
    for h in range(env.height):
        for w in range(env.width + 1):
            if env.walls[0][h, w] == 1:
                ax.axvline(x=w, ymin=h/env.height, ymax=(h+1)/env.height, 
                          color='black', linewidth=3)
    
    # Dessiner les murs horizontaux
    for h in range(env.height + 1):
        for w in range(env.width):
            if env.walls[1][h, w] == 1:
                ax.axhline(y=h, xmin=w/env.width, xmax=(w+1)/env.width, 
                          color='black', linewidth=3)
    
    # Marquer la position de départ (vert)
    start_pos = (0, 0)
    ax.add_patch(plt.Rectangle((start_pos[1], env.height - start_pos[0] - 1), 1, 1, 
                              facecolor='lightgreen', alpha=0.7))
    
    # Marquer la position d'arrivée (rouge)
    goal_pos = (env.height - 1, env.width - 1)
    ax.add_patch(plt.Rectangle((goal_pos[1], env.height - goal_pos[0] - 1), 1, 1, 
                              facecolor='lightcoral', alpha=0.7))
    
    # Dessiner le parcours
    if len(path) > 1:
        # Convertir les coordonnées pour l'affichage (inverser y)
        path_x = [pos[1] + 0.5 for pos in path]
        path_y = [env.height - pos[0] - 0.5 for pos in path]
        
        ax.plot(path_x, path_y, 'b-', linewidth=2, alpha=0.7, label='Parcours')
        ax.plot(path_x[0], path_y[0], 'go', markersize=8, label='Début')
        ax.plot(path_x[-1], path_y[-1], 'ro', markersize=8, label='Fin')
    
    ax.set_xlim(0, env.width)
    ax.set_ylim(0, env.height)
    ax.set_aspect('equal')
    ax.set_title(f'Épisode {episode_num + 1} - Parcours ({len(path)-1} pas)')
    ax.legend()
    ax.grid(True, alpha=0.3)

# Créer la figure interactive avec slider
fig, ax = plt.subplots(figsize=(12, 8))
plt.subplots_adjust(bottom=0.15)

# Position initiale du slider
initial_episode = 0
draw_maze_with_path(ax, env, parcours[initial_episode], initial_episode)

# Créer le slider
ax_slider = plt.axes([0.1, 0.05, 0.8, 0.03])
slider = Slider(ax_slider, 'Épisode', 0, nb_episodes-1, 
                valinit=initial_episode, valfmt='%d')

# Fonction de mise à jour du slider
def update_episode(val):
    episode = int(slider.val)
    draw_maze_with_path(ax, env, parcours[episode], episode)
    fig.canvas.draw()

slider.on_changed(update_episode)

plt.show()
