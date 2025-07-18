"""
Example comparing Q-learning and SARSA algorithms.

This script compares the performance of Q-learning and SARSA algorithms
on the same maze environment and displays the results.
"""

import sys
import os

# Add src folder to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from maze import Test_maze_with_traps
from agent_epsilon_greedy import Agent_epsilon_greedy
import matplotlib.pyplot as plt
import numpy as np


def train_and_evaluate(env, method='q_learning', num_episodes=1000, epsilon=0.1):
    """
    Train an agent with the specified method and return scores.
    
    Args:
        env: Maze environment
        method: 'q_learning' or 'sarsa'
        num_episodes: Number of training episodes
        epsilon: Exploration rate
    
    Returns:
        step_scores: List of step counts per episode
        reward_scores: List of total rewards per episode
    """
    agent = Agent_epsilon_greedy(
        height=env.height,
        width=env.width,
        epsilon=epsilon,
        alpha=0.1,
        gamma=0.9
    )
    
    step_scores = []
    reward_scores = []
    
    for episode in range(num_episodes):
        env.reset()
        done = False
        state = env.agent_pos
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            
            # Handle impossible moves
            while next_state is None:
                action = agent.select_action(state)
                next_state, reward, done = env.step(action)
            
            agent.update(state, action, reward, next_state, done, method=method)
            state = next_state
        
        step_scores.append(env.step_numbers)
        reward_scores.append(env.total_reward)
    
    return step_scores, reward_scores


def main():
    """Main comparison function."""
    print("=== Q-learning vs SARSA Comparison ===\n")
    
    # Create environment (maze with traps)
    env = Test_maze_with_traps()
    print(f"Environment: {env.width}x{env.height} with traps")
    print(f"Start position: {env.start_position}")
    print(f"Goal position: {env.end_position}\n")
    
    num_episodes = 1000
    num_runs = 5  # Number of runs to average results
    
    print(f"Training for {num_episodes} episodes, {num_runs} runs per algorithm...\n")
    
    # Store results
    results_q_learning = []
    results_sarsa = []
    
    # Run each algorithm multiple times
    for run in range(num_runs):
        print(f"Run {run + 1}/{num_runs}...")
        
        # Q-learning
        print("  - Q-learning...")
        step_scores_q, reward_scores_q = train_and_evaluate(env, method='q_learning', num_episodes=num_episodes)
        results_q_learning.append([step_scores_q, reward_scores_q])
        
        # SARSA
        print("  - SARSA...")
        step_scores_sarsa, reward_scores_sarsa = train_and_evaluate(env, method='sarsa', num_episodes=num_episodes)
        results_sarsa.append([step_scores_sarsa, reward_scores_sarsa])
    
    # Separate step and reward results
    step_results_q = [run[0] for run in results_q_learning]
    reward_results_q = [run[1] for run in results_q_learning]
    step_results_sarsa = [run[0] for run in results_sarsa]
    reward_results_sarsa = [run[1] for run in results_sarsa]
    
    # Calculate averages
    mean_steps_q = np.mean(step_results_q, axis=0)
    mean_rewards_q = np.mean(reward_results_q, axis=0)
    mean_steps_sarsa = np.mean(step_results_sarsa, axis=0)
    mean_rewards_sarsa = np.mean(reward_results_sarsa, axis=0)
    
    std_steps_q = np.std(step_results_q, axis=0)
    std_rewards_q = np.std(reward_results_q, axis=0)
    std_steps_sarsa = np.std(step_results_sarsa, axis=0)
    std_rewards_sarsa = np.std(reward_results_sarsa, axis=0)
    
    # Analyze results
    print("\n=== Results Analysis ===")
    
    # Final scores (last 100 episodes)
    final_steps_q = np.mean(mean_steps_q[-100:])
    final_steps_sarsa = np.mean(mean_steps_sarsa[-100:])
    final_rewards_q = np.mean(mean_rewards_q[-100:])
    final_rewards_sarsa = np.mean(mean_rewards_sarsa[-100:])
    
    print(f"Final average performance (last 100 episodes):")
    print(f"  Q-learning - Steps: {final_steps_q:.2f} ± {np.mean(std_steps_q[-100:]):.2f}")
    print(f"  Q-learning - Reward: {final_rewards_q:.2f} ± {np.mean(std_rewards_q[-100:]):.2f}")
    print(f"  SARSA - Steps: {final_steps_sarsa:.2f} ± {np.mean(std_steps_sarsa[-100:]):.2f}")
    print(f"  SARSA - Reward: {final_rewards_sarsa:.2f} ± {np.mean(std_rewards_sarsa[-100:]):.2f}")
    
    # Convergence (episode where score becomes stable)
    def find_convergence(scores, threshold=0.1):
        """Find approximate convergence episode."""
        window = 100
        if len(scores) < window * 2:
            return len(scores)
        
        for i in range(window, len(scores) - window):
            before = np.mean(scores[i-window:i])
            after = np.mean(scores[i:i+window])
            if abs(before - after) / before < threshold:
                return i
        return len(scores)
    
    conv_steps_q = find_convergence(mean_steps_q)
    conv_steps_sarsa = find_convergence(mean_steps_sarsa)
    
    print(f"\nApproximate convergence (steps):")
    print(f"  Q-learning: episode {conv_steps_q}")
    print(f"  SARSA:      episode {conv_steps_sarsa}")
    
    # Visualization
    print("\n=== Results Visualization ===")
    
    plt.figure(figsize=(20, 12))
    
    # Graph 1: Steps comparison
    plt.subplot(2, 3, 1)
    episodes = range(len(mean_steps_q))
    
    plt.plot(episodes, mean_steps_q, 'b-', label='Q-learning', alpha=0.8)
    plt.fill_between(episodes, 
                     mean_steps_q - std_steps_q, 
                     mean_steps_q + std_steps_q, 
                     alpha=0.2, color='blue')
    
    plt.plot(episodes, mean_steps_sarsa, 'r-', label='SARSA', alpha=0.8)
    plt.fill_between(episodes, 
                     mean_steps_sarsa - std_steps_sarsa, 
                     mean_steps_sarsa + std_steps_sarsa, 
                     alpha=0.2, color='red')
    
    plt.title('Steps Comparison')
    plt.xlabel('Episode')
    plt.ylabel('Number of Steps')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Graph 2: Rewards comparison
    plt.subplot(2, 3, 2)
    plt.plot(episodes, mean_rewards_q, 'b-', label='Q-learning', alpha=0.8)
    plt.fill_between(episodes, 
                     mean_rewards_q - std_rewards_q, 
                     mean_rewards_q + std_rewards_q, 
                     alpha=0.2, color='blue')
    
    plt.plot(episodes, mean_rewards_sarsa, 'r-', label='SARSA', alpha=0.8)
    plt.fill_between(episodes, 
                     mean_rewards_sarsa - std_rewards_sarsa, 
                     mean_rewards_sarsa + std_rewards_sarsa, 
                     alpha=0.2, color='red')
    
    plt.title('Total Rewards Comparison')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward per Episode')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Graph 3: Moving averages for steps
    plt.subplot(2, 3, 3)
    window = 50
    
    if len(mean_steps_q) >= window:
        smooth_steps_q = np.convolve(mean_steps_q, np.ones(window)/window, mode='valid')
        smooth_steps_sarsa = np.convolve(mean_steps_sarsa, np.ones(window)/window, mode='valid')
        
        plt.plot(range(window-1, len(mean_steps_q)), smooth_steps_q, 'b-', label='Q-learning', linewidth=2)
        plt.plot(range(window-1, len(mean_steps_sarsa)), smooth_steps_sarsa, 'r-', label='SARSA', linewidth=2)
    
    plt.title(f'Steps Moving Averages (window = {window})')
    plt.xlabel('Episode')
    plt.ylabel('Number of Steps')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Graph 4: Moving averages for rewards
    plt.subplot(2, 3, 4)
    if len(mean_rewards_q) >= window:
        smooth_rewards_q = np.convolve(mean_rewards_q, np.ones(window)/window, mode='valid')
        smooth_rewards_sarsa = np.convolve(mean_rewards_sarsa, np.ones(window)/window, mode='valid')
        
        plt.plot(range(window-1, len(mean_rewards_q)), smooth_rewards_q, 'b-', label='Q-learning', linewidth=2)
        plt.plot(range(window-1, len(mean_rewards_sarsa)), smooth_rewards_sarsa, 'r-', label='SARSA', linewidth=2)
    
    plt.title(f'Rewards Moving Averages (window = {window})')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Graph 5: Final performance distribution
    plt.subplot(2, 3, 5)
    final_steps_q_all = [np.mean(run[-100:]) for run in step_results_q]
    final_steps_sarsa_all = [np.mean(run[-100:]) for run in step_results_sarsa]
    final_rewards_q_all = [np.mean(run[-100:]) for run in reward_results_q]
    final_rewards_sarsa_all = [np.mean(run[-100:]) for run in reward_results_sarsa]
    
    plt.boxplot([final_steps_q_all, final_steps_sarsa_all], 
                labels=['Q-learning', 'SARSA'])
    plt.title('Final Steps Distribution')
    plt.ylabel('Number of Steps (last 100 episodes)')
    plt.grid(True, alpha=0.3)
    
    # Graph 6: Final rewards distribution
    plt.subplot(2, 3, 6)
    plt.boxplot([final_rewards_q_all, final_rewards_sarsa_all], 
                labels=['Q-learning', 'SARSA'])
    plt.title('Final Rewards Distribution')
    plt.ylabel('Total Reward (last 100 episodes)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Conclusion
    print(f"\n=== Conclusion ===")
    print("Steps performance:")
    if final_steps_q < final_steps_sarsa:
        print("  Q-learning requires fewer steps (better)")
    elif final_steps_sarsa < final_steps_q:
        print("  SARSA requires fewer steps (better)")
    else:
        print("  Both algorithms have similar step performance")
    
    print("Reward performance:")
    if final_rewards_q > final_rewards_sarsa:
        print("  Q-learning achieves higher rewards (better)")
    elif final_rewards_sarsa > final_rewards_q:
        print("  SARSA achieves higher rewards (better)")
    else:
        print("  Both algorithms have similar reward performance")
    
    print(f"\nPerformance differences:")
    print(f"  Steps: {abs(final_steps_q - final_steps_sarsa):.2f} steps")
    print(f"  Rewards: {abs(final_rewards_q - final_rewards_sarsa):.2f} points")


if __name__ == "__main__":
    main()
