"""
Main script for training a Q-learning agent in a maze environment.

This script initializes a maze environment with traps, creates an epsilon-greedy agent,
and trains it over a defined number of episodes. It then displays the results as a graph
and allows visualization of the paths taken by the agent.
"""

from maze import Maze, Test_maze, Test_maze_little, Test_maze_with_traps
from agent_epsilon_greedy import Agent_epsilon_greedy
import config
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider
import pickle
import datetime


def main():
    """
    Main function that orchestrates agent training and results display.
    """
    # Initialize maze environment with traps
    env = Test_maze_with_traps()
    env.draw()  # Initial maze display
    
    # Create epsilon-greedy agent with learning parameters
    agent = Agent_epsilon_greedy(
        height=env.height,
        width=env.width,
        epsilon=0.1,    # Initial exploration rate
        alpha=0.1,      # Learning rate
        gamma=0.9       # Discount factor
    )

    # Training parameters
    num_episodes = 1000
    scores = []      # Store scores (number of steps) per episode
    epsilons = []    # Store epsilon values per episode
    paths = []       # Store paths taken per episode

    print("Starting training...")
    
    # Main training loop
    for episode in tqdm(range(num_episodes)):
        env.reset()  # Reset environment
        path = []    # Path for this episode
        
        done = False
        state = env.agent_pos  # Initial agent position
        path.append(state)
        
        # Episode loop
        while not done:
            # Select action according to epsilon-greedy policy
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            
            # Handle invalid moves (wall collision)
            while next_state == None:
                action = agent.select_action(state)
                next_state, reward, done = env.step(action)
            
            # Update Q-table with SARSA algorithm
            agent.update(state, action, reward, next_state, done, method='sarsa')
            state = next_state
            path.append(state)
            
        paths.append(path)
            
        # Record performance metrics
        final_score = env.step_numbers  # Number of steps to reach goal
        scores.append(final_score)
        epsilons.append(agent.epsilon)
        
        # Epsilon decay (currently disabled)
        # agent.decay_epsilon()
        
    print(f"Training completed. Average score of last 50 episodes: {np.mean(scores[-50:]):.2f}")

    # Optional saving of paths taken
    # with open("paths.pkl", 'wb') as f:
    #     pickle.dump(paths, f)

    # Graphical display of score evolution
    plt.figure(figsize=(12, 5))
    plt.plot(scores)
    plt.title('Score Evolution per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Number of Steps')
    plt.show()

    # Interactive visualization of maze with paths taken
    config.draw_maze_with_path(env, paths, grid=True)


if __name__ == "__main__":
    main()
