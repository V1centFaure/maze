"""
Example of creating and using a custom maze.

This script demonstrates how to create a custom maze environment
with specific walls and reward structure, then train an agent on it.
"""

import sys
import os

# Add src folder to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from maze import Maze
from agent_epsilon_greedy import Agent_epsilon_greedy
import config
import numpy as np
import matplotlib.pyplot as plt


def create_custom_maze():
    """
    Create a custom maze with specific layout and rewards.
    
    Returns:
        Maze: Custom maze instance
    """
    # Define maze dimensions
    width, height = 6, 4
    
    # Create wall matrices
    # Vertical walls (between columns)
    walls_vertical = np.zeros((height, width + 1))
    # Horizontal walls (between rows)
    walls_horizontal = np.zeros((height + 1, width))
    
    # Set boundary walls
    walls_vertical[:, [0, -1]] = 1  # Left and right boundaries
    walls_horizontal[[0, -1], :] = 1  # Top and bottom boundaries
    
    # Add internal walls to create interesting paths
    walls_vertical[1, 2] = 1  # Vertical wall at row 1, column 2
    walls_vertical[2, 4] = 1  # Vertical wall at row 2, column 4
    walls_vertical[0, 3] = 1  # Vertical wall at row 0, column 3
    
    walls_horizontal[2, 1] = 1  # Horizontal wall at row 2, column 1
    walls_horizontal[1, 3] = 1  # Horizontal wall at row 1, column 3
    walls_horizontal[3, 4] = 1  # Horizontal wall at row 3, column 4
    
    # Create custom reward grid
    reward_grid = np.array([
        [-1, -1, -1, -5, -1, -1],  # Row 0: normal cells with one trap
        [-1, -2, -1, -1, -1, -1],  # Row 1: small penalty cell
        [-1, -1, -1, -1, -10, -1], # Row 2: big trap
        [-1, -1, -1, -1, -1, 100]  # Row 3: goal with high reward
    ])
    
    # Create the maze
    custom_maze = Maze(
        width=width,
        height=height,
        walls=[walls_vertical, walls_horizontal],
        grid=reward_grid,
        start_position=(0, 0),  # Top-left corner
        end_position=(3, 5),    # Bottom-right corner
        cell_size=100
    )
    
    return custom_maze


def train_on_custom_maze(env, num_episodes=800):
    """
    Train an agent on the custom maze.
    
    Args:
        env: Custom maze environment
        num_episodes: Number of training episodes
    
    Returns:
        agent: Trained agent
        step_scores: Training step scores
        reward_scores: Training reward scores
        paths: Training paths
    """
    # Create agent
    agent = Agent_epsilon_greedy(
        height=env.height,
        width=env.width,
        epsilon=0.15,  # Higher exploration for complex maze
        alpha=0.1,
        gamma=0.95     # Higher discount factor for long-term planning
    )
    
    step_scores = []
    reward_scores = []
    paths = []
    
    print(f"Training agent on custom maze for {num_episodes} episodes...")
    
    for episode in range(num_episodes):
        env.reset()
        path = [env.agent_pos]
        done = False
        state = env.agent_pos
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            
            # Handle wall collisions
            while next_state is None:
                action = agent.select_action(state)
                next_state, reward, done = env.step(action)
            
            # Update agent with Q-learning
            agent.update(state, action, reward, next_state, done, method='q_learning')
            state = next_state
            path.append(state)
        
        step_scores.append(env.step_numbers)
        reward_scores.append(env.total_reward)
        paths.append(path)
        
        # Show progress
        if (episode + 1) % 100 == 0:
            avg_steps = np.mean(step_scores[-100:])
            avg_reward = np.mean(reward_scores[-100:])
            print(f"Episode {episode + 1}: Avg steps = {avg_steps:.2f}, Avg reward = {avg_reward:.2f}")
    
    return agent, step_scores, reward_scores, paths


def analyze_custom_maze_results(env, agent, step_scores, reward_scores, paths):
    """
    Analyze and visualize results from custom maze training.
    
    Args:
        env: Maze environment
        agent: Trained agent
        step_scores: Training step scores
        reward_scores: Training reward scores
        paths: Training paths
    """
    print("\n=== Custom Maze Analysis ===")
    
    # Performance analysis
    initial_steps = np.mean(step_scores[:50])
    final_steps = np.mean(step_scores[-50:])
    best_steps = min(step_scores)
    
    initial_rewards = np.mean(reward_scores[:50])
    final_rewards = np.mean(reward_scores[-50:])
    best_rewards = max(reward_scores)
    
    print(f"Steps performance:")
    print(f"  Initial (first 50 episodes): {initial_steps:.2f}")
    print(f"  Final (last 50 episodes): {final_steps:.2f}")
    print(f"  Best achieved: {best_steps}")
    print(f"  Improvement: {initial_steps - final_steps:.2f} steps")
    
    print(f"\nReward performance:")
    print(f"  Initial (first 50 episodes): {initial_rewards:.2f}")
    print(f"  Final (last 50 episodes): {final_rewards:.2f}")
    print(f"  Best achieved: {best_rewards}")
    print(f"  Improvement: {final_rewards - initial_rewards:.2f} points")
    
    # Analyze Q-table
    print(f"\nQ-table statistics:")
    print(f"  Max Q-value: {np.max(agent.q_table):.2f}")
    print(f"  Min Q-value: {np.min(agent.q_table):.2f}")
    print(f"  Average Q-value: {np.mean(agent.q_table):.2f}")
    
    # Find optimal policy
    print(f"\nOptimal policy (best action per state):")
    for row in range(env.height):
        policy_row = []
        for col in range(env.width):
            best_action_idx = np.argmax(agent.q_table[row, col, :])
            best_action = config.IDX2ACTION[best_action_idx]
            policy_row.append(best_action[:1].upper())  # First letter
        print(f"  Row {row}: {' '.join(policy_row)}")
    
    # Visualization
    plt.figure(figsize=(18, 12))
    
    # Plot 1: Steps training curve
    plt.subplot(3, 3, 1)
    plt.plot(step_scores, alpha=0.7, color='blue')
    
    # Add moving average
    window = 50
    if len(step_scores) >= window:
        moving_avg = np.convolve(step_scores, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(step_scores)), moving_avg, 'r-', linewidth=2)
    
    plt.title('Steps Training Progress')
    plt.xlabel('Episode')
    plt.ylabel('Steps to Goal')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Rewards training curve
    plt.subplot(3, 3, 2)
    plt.plot(reward_scores, alpha=0.7, color='green')
    
    if len(reward_scores) >= window:
        moving_avg_reward = np.convolve(reward_scores, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(reward_scores)), moving_avg_reward, 'r-', linewidth=2)
    
    plt.title('Rewards Training Progress')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Steps vs Rewards correlation
    plt.subplot(3, 3, 3)
    plt.scatter(step_scores, reward_scores, alpha=0.6, s=10)
    plt.xlabel('Steps')
    plt.ylabel('Total Reward')
    plt.title('Steps vs Rewards Correlation')
    plt.grid(True, alpha=0.3)
    
    # Plots 4-7: Q-values heatmap for each action
    actions = ['Up', 'Down', 'Left', 'Right']
    for i, action in enumerate(actions):
        plt.subplot(3, 3, i+4)
        q_values_action = agent.q_table[:, :, i]
        plt.imshow(q_values_action, cmap='viridis', aspect='auto')
        plt.title(f'Q-values: {action}')
        plt.colorbar()
        
        # Add text annotations
        for row in range(env.height):
            for col in range(env.width):
                plt.text(col, row, f'{q_values_action[row, col]:.1f}', 
                        ha='center', va='center', color='white', fontsize=8)
    
    # Plot 8: Path length distribution
    plt.subplot(3, 3, 8)
    path_lengths = [len(path) for path in paths]
    plt.hist(path_lengths, bins=20, alpha=0.7, edgecolor='black')
    plt.title('Path Length Distribution')
    plt.xlabel('Path Length')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # Plot 9: Performance evolution comparison
    plt.subplot(3, 3, 9)
    episodes = range(len(step_scores))
    
    # Normalize both metrics to 0-1 scale for comparison
    norm_steps = [(s - min(step_scores)) / (max(step_scores) - min(step_scores)) for s in step_scores]
    norm_rewards = [(r - min(reward_scores)) / (max(reward_scores) - min(reward_scores)) for r in reward_scores]
    
    plt.plot(episodes, norm_steps, alpha=0.7, color='blue', label='Steps (normalized)')
    plt.plot(episodes, norm_rewards, alpha=0.7, color='green', label='Rewards (normalized)')
    plt.xlabel('Episode')
    plt.ylabel('Normalized Performance')
    plt.title('Performance Evolution Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def main():
    """Main function for custom maze demonstration."""
    print("=== Custom Maze Example ===\n")
    
    # 1. Create custom maze
    print("1. Creating custom maze...")
    env = create_custom_maze()
    print(f"   Custom maze created: {env.width}x{env.height}")
    print(f"   Start: {env.start_position}, Goal: {env.end_position}")
    print(f"   Reward structure includes traps and penalties\n")
    
    # 2. Display maze structure
    print("2. Maze reward structure:")
    for row in range(env.height):
        row_str = "   "
        for col in range(env.width):
            reward = env.grid[row, col]
            if reward == 100:
                row_str += "GOAL "
            elif reward < -5:
                row_str += "TRAP "
            elif reward < -1:
                row_str += "PEN  "
            else:
                row_str += "NORM "
        print(row_str)
    print()
    
    # 3. Train agent
    print("3. Training agent on custom maze...")
    agent, step_scores, reward_scores, paths = train_on_custom_maze(env, num_episodes=800)
    
    # 4. Analyze results
    analyze_custom_maze_results(env, agent, step_scores, reward_scores, paths)
    
    # 5. Save trained model
    print("\n5. Saving trained model...")
    agent.save_q_table("custom_maze_agent.pkl")
    
    # 6. Interactive visualization
    print("\n6. Interactive maze visualization...")
    print("   Close the maze window to continue...")
    env.draw(draw_value=True, q_values=np.max(agent.q_table, axis=2), grid=True)
    
    # 7. Path visualization
    print("\n7. Interactive path visualization...")
    print("   Use slider to see learning progression.")
    print("   Close window to end program.")
    config.draw_maze_with_path(env, paths, grid=True)
    
    print("\n=== Custom maze demonstration completed ===")


if __name__ == "__main__":
    main()
