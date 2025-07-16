from maze import Maze, Test_maze, Test_maze_little
from agent_epsilon_greedy import Agent_epsilon_greedy
import config
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider
import pickle
import datetime


env = Test_maze()
env.draw()
agent = Agent_epsilon_greedy(height = env.height,
                             width= env.width,
                             epsilon=0.3, 
                             alpha=0.1, 
                             gamma=0.9)

nb_episodes = 200
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


# with open("paths.pkl", 'wb') as f:
#             pickle.dump(parcours, f)


# Affichage des scores
plt.figure(figsize=(12, 5))
plt.plot(scores)
plt.title('Évolution des scores par épisode')
plt.xlabel('Épisode')
plt.ylabel('Nombre de pas')
plt.show()

config.draw_maze_with_path(env, parcours)


