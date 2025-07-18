"""
MAZE - Reinforcement Learning Library

A comprehensive Python library for reinforcement learning in maze environments
with interactive visualization capabilities.

This package provides:
- Customizable maze environments with walls, rewards, and traps
- Reinforcement learning agents using Q-learning and SARSA algorithms
- Interactive visualization with Pygame for real-time training observation
- Episode navigation with sliders for learning evolution analysis
- Epsilon-greedy policy with automatic decay
- Q-table persistence for model saving/loading
- Predefined mazes of different complexities for testing

Main Classes:
    Maze: Base maze environment class
    Test_maze: Medium-sized test maze (8×5)
    Test_maze_little: Small test maze (4×3) 
    Test_maze_with_traps: Complex maze with traps and variable rewards
    Agent_epsilon_greedy: Reinforcement learning agent with epsilon-greedy policy

Example:
    >>> from maze import Test_maze
    >>> from agent_epsilon_greedy import Agent_epsilon_greedy
    >>> 
    >>> # Create environment and agent
    >>> env = Test_maze()
    >>> agent = Agent_epsilon_greedy(env.height, env.width)
    >>> 
    >>> # Train the agent
    >>> for episode in range(1000):
    ...     env.reset()
    ...     # ... training loop ...
    >>> 
    >>> # Visualize results
    >>> env.draw()

Author: V1centFaure
License: MIT
"""

from .maze import Maze, Test_maze, Test_maze_little, Test_maze_with_traps
from .agent_epsilon_greedy import Agent_epsilon_greedy
from . import config

__version__ = "1.0.0"
__author__ = "V1centFaure"
__email__ = ""
__license__ = "MIT"

__all__ = [
    "Maze",
    "Test_maze", 
    "Test_maze_little",
    "Test_maze_with_traps",
    "Agent_epsilon_greedy",
    "config"
]
