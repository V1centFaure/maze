"""
Example of using the MAZE library after installation via pip.

This script demonstrates how to import and use the library
after it has been installed with:
    pip install git+https://github.com/V1centFaure/maze.git

Usage:
    python pip_usage_example.py
"""

# After pip installation, import like this:
from maze import Test_maze, Agent_epsilon_greedy
import maze.config as config
import matplotlib.pyplot as plt
import numpy as np


def main():
    """Demonstration of the MAZE library after pip installation."""
    print("=== MAZE Library - Post-Installation Example ===\n")
    
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
        epsilon=0.1,
        alpha=0.1,
        gamma=0.9
    )
    print(f"   Agent created with epsilon={agent.epsilon}\n")
    
    # 3. Quick training demonstration
    print("3. Quick training (100 episodes)...")
    step_scores = []
    
    for episode in range(100):
        env.reset()
        done = False
        state = env.agent_pos
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            
            # Handle wall collisions
            while next_state is None:
                action = agent.select_action(state)
                next_state, reward, done = env.step(action)
            
            agent.update(state, action, reward, next_state, done, method='q_learning')
            state = next_state
        
        step_scores.append(env.step_numbers)
        
        if (episode + 1) % 25 == 0:
            avg_steps = np.mean(step_scores[-25:])
            print(f"   Episode {episode + 1}: Avg steps = {avg_steps:.2f}")
    
    # 4. Results
    print(f"\n4. Training completed!")
    print(f"   Initial performance (first 10): {np.mean(step_scores[:10]):.2f} steps")
    print(f"   Final performance (last 10): {np.mean(step_scores[-10:]):.2f} steps")
    print(f"   Best performance: {min(step_scores)} steps")
    
    # 5. Visualize the maze
    print("\n5. Displaying the maze...")
    print("   Close the window to end the program.")
    env.draw()
    
    print("\n=== Example completed successfully ===")


if __name__ == "__main__":
    main()
