"""
Basic usage example of the MAZE library.

This script demonstrates how to create a maze environment,
train an agent with Q-learning, and visualize the results.
"""

import sys
import os

# Add src folder to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from maze import Test_maze
from agent_epsilon_greedy import Agent_epsilon_greedy
import config
import matplotlib.pyplot as plt
import numpy as np


def train_agent(env, agent, num_episodes=500, method='q_learning'):
    """
    Train an agent in the given environment.
    
    Args:
        env: Maze environment
        agent: Agent to train
        num_episodes: Number of training episodes
        method: Learning method ('q_learning' or 'sarsa')
    
    Returns:
        step_scores: List of step counts per episode
        reward_scores: List of total rewards per episode
        paths: List of paths taken per episode
    """
    step_scores = []
    reward_scores = []
    paths = []
    
    print(f"Training with {method} for {num_episodes} episodes...")
    
    for episode in range(num_episodes):
        env.reset()
        path = [env.agent_pos]
        done = False
        state = env.agent_pos
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            
            # Handle impossible moves (wall collision)
            while next_state is None:
                action = agent.select_action(state)
                next_state, reward, done = env.step(action)
            
            # Update the agent
            agent.update(state, action, reward, next_state, done, method=method)
            state = next_state
            path.append(state)
        
        step_scores.append(env.step_numbers)
        reward_scores.append(env.total_reward)
        paths.append(path)
        
        # Show progress every 100 episodes
        if (episode + 1) % 100 == 0:
            avg_steps = np.mean(step_scores[-100:])
            avg_reward = np.mean(reward_scores[-100:])
            print(f"Episode {episode + 1}: Avg steps = {avg_steps:.2f}, Avg reward = {avg_reward:.2f}")
    
    return step_scores, reward_scores, paths


def main():
    """Main demonstration function."""
    print("=== MAZE Library Demonstration ===\n")
    
    # 1. Create the environment
    print("1. Creating maze environment...")
    env = Test_maze()
    print(f"   Maze created: {env.width}x{env.height}")
    print(f"   Start position: {env.start_position}")
    print(f"   Goal position: {env.end_position}\n")
    
    # 2. Create the agent
    print("2. Creating epsilon-greedy agent...")
    agent = Agent_epsilon_greedy(
        height=env.height,
        width=env.width,
        epsilon=0.1,    # 10% exploration
        alpha=0.1,      # Learning rate
        gamma=0.9       # Discount factor
    )
    print(f"   Agent created with epsilon={agent.epsilon}, alpha={agent.alpha}, gamma={agent.gamma}\n")
    
    # 3. Train the agent
    print("3. Training the agent...")
    step_scores, reward_scores, paths = train_agent(env, agent, num_episodes=1000, method='q_learning')
    
    # 4. Analyze results
    print("\n4. Results analysis:")
    print(f"   Steps - Initial (first 10 episodes): {np.mean(step_scores[:10]):.2f}")
    print(f"   Steps - Final (last 50 episodes): {np.mean(step_scores[-50:]):.2f}")
    print(f"   Steps - Best: {min(step_scores)}")
    print(f"   Steps - Overall average: {np.mean(step_scores):.2f}")
    print(f"   Reward - Initial (first 10 episodes): {np.mean(reward_scores[:10]):.2f}")
    print(f"   Reward - Final (last 50 episodes): {np.mean(reward_scores[-50:]):.2f}")
    print(f"   Reward - Best: {max(reward_scores)}")
    print(f"   Reward - Overall average: {np.mean(reward_scores):.2f}")
    
    # 5. Visualize performance evolution
    print("\n5. Displaying evolution graphs...")
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Steps evolution
    plt.subplot(2, 2, 1)
    plt.plot(step_scores, alpha=0.7, color='blue')
    
    # Add moving average to smooth the curve
    window_size = 50
    if len(step_scores) >= window_size:
        moving_avg = np.convolve(step_scores, np.ones(window_size)/window_size, mode='valid')
        plt.plot(range(window_size-1, len(step_scores)), moving_avg, 'r-', linewidth=2, label=f'Moving average ({window_size} episodes)')
        plt.legend()
    
    plt.title('Steps Evolution During Training')
    plt.xlabel('Episode')
    plt.ylabel('Number of Steps to Reach Goal')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Reward evolution
    plt.subplot(2, 2, 2)
    plt.plot(reward_scores, alpha=0.7, color='green')
    
    if len(reward_scores) >= window_size:
        moving_avg_reward = np.convolve(reward_scores, np.ones(window_size)/window_size, mode='valid')
        plt.plot(range(window_size-1, len(reward_scores)), moving_avg_reward, 'r-', linewidth=2, label=f'Moving average ({window_size} episodes)')
        plt.legend()
    
    plt.title('Total Reward Evolution During Training')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward per Episode')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Steps vs Reward correlation
    plt.subplot(2, 2, 3)
    plt.scatter(step_scores, reward_scores, alpha=0.6, s=10)
    plt.xlabel('Number of Steps')
    plt.ylabel('Total Reward')
    plt.title('Steps vs Reward Correlation')
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Performance improvement over time
    plt.subplot(2, 2, 4)
    episodes = range(len(step_scores))
    plt.plot(episodes, step_scores, alpha=0.5, color='blue', label='Steps')
    
    # Normalize reward scores to same scale for comparison
    normalized_rewards = [(r - min(reward_scores)) / (max(reward_scores) - min(reward_scores)) * (max(step_scores) - min(step_scores)) + min(step_scores) for r in reward_scores]
    plt.plot(episodes, normalized_rewards, alpha=0.5, color='green', label='Normalized Rewards')
    
    plt.xlabel('Episode')
    plt.ylabel('Performance Metric')
    plt.title('Combined Performance Evolution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 6. Save the trained model
    print("\n6. Saving the model...")
    agent.save_q_table("example_trained_agent.pkl")
    
    # 7. Visualize the maze
    print("\n7. Displaying the maze...")
    print("   Close the maze window to continue...")
    env.draw()
    
    # 8. Interactive path visualization
    print("\n8. Interactive path visualization...")
    print("   Use the slider to navigate between episodes.")
    print("   Close the window to end the program.")
    config.draw_maze_with_path(env, paths, grid=True)
    
    print("\n=== Demonstration completed ===")


if __name__ == "__main__":
    main()
