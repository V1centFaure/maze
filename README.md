# MAZE - Reinforcement Learning Library

A comprehensive Python library for reinforcement learning in maze environments with interactive visualization.

## 🎯 Features

- **Customizable maze environments** with walls, rewards, and traps
- **Reinforcement learning agents** using Q-learning and SARSA algorithms
- **Interactive visualization** with Pygame for real-time training observation
- **Episode navigation** with sliders for learning evolution analysis
- **Epsilon-greedy policy** with automatic decay
- **Q-table persistence** for model saving/loading
- **Predefined mazes** of different complexities for testing

## 🚀 Installation

### Installation from GitHub

```bash
# Direct installation from GitHub
pip install git+https://github.com/V1centFaure/maze.git

# Or clone and install locally
git clone https://github.com/V1centFaure/maze.git
cd maze
pip install -e .
```

### Dependencies

The library requires the following packages:
- `numpy>=1.20.0` - Numerical computations
- `pygame>=2.0.0` - Interactive visualization
- `matplotlib>=3.3.0` - Plotting and analysis
- `tqdm>=4.60.0` - Progress bars

These dependencies are automatically installed with the package.

## 📖 Quick Start

### Basic Example

```python
# After pip installation, import like this:
from maze import Test_maze, Agent_epsilon_greedy
import maze.config as config

# Create a test maze
env = Test_maze()

# Create an epsilon-greedy agent
agent = Agent_epsilon_greedy(
    height=env.height,
    width=env.width,
    epsilon=0.1,    # Exploration rate
    alpha=0.1,      # Learning rate
    gamma=0.9       # Discount factor
)

# Train the agent
for episode in range(1000):
    env.reset()
    done = False
    state = env.agent_pos
    
    while not done:
        action = agent.select_action(state)
        next_state, reward, done = env.step(action)
        
        if next_state is not None:
            agent.update(state, action, reward, next_state, done, method='q_learning')
            state = next_state

# Visualize the maze
env.draw()
```

### Example with Path Visualization

```python
from maze import Test_maze_with_traps, Agent_epsilon_greedy
import maze.config as config

# Create a maze with traps
env = Test_maze_with_traps()

# Train agent and collect paths
paths = []
for episode in range(100):
    env.reset()
    path = [env.agent_pos]
    done = False
    
    while not done:
        action = agent.select_action(env.agent_pos)
        next_state, reward, done = env.step(action)
        if next_state is not None:
            path.append(next_state)
            agent.update(env.agent_pos, action, reward, next_state, done)

    paths.append(path)

# Interactive path visualization
config.draw_maze_with_path(env, paths, grid=True)
```

## 🏗️ Architecture

### Main Classes

#### `Maze`
Base class for maze environments:
- Wall and reward grid management
- Collision detection and agent movement
- Pygame visualization
- Support for Q-value display

#### `Agent_epsilon_greedy`
Reinforcement learning agent:
- Epsilon-greedy policy for exploration/exploitation
- Q-learning and SARSA support
- Automatic epsilon decay
- Q-table saving/loading

#### Predefined Mazes
- `Test_maze`: Medium maze (8×5) for standard tests
- `Test_maze_little`: Small maze (4×3) for quick tests
- `Test_maze_with_traps`: Complex maze with traps and variable rewards

### `config` Module
Centralized configuration:
- Color and display constants
- Reward parameters
- Action mappings
- Utility functions (argmax with tie-breaking)

## 🎮 Command Line Usage

After installation, you can run a demonstration:

```bash
maze-demo
```

## 📊 Advanced Features

### Model Saving and Loading

```python
# Save a trained model
agent.save_q_table("my_trained_agent.pkl")

# Load an existing model
agent.load_q_table("my_trained_agent.pkl")
```

### Custom Maze Creation

```python
import numpy as np
from maze import Maze

# Create a custom maze
width, height = 6, 4
walls_vertical = np.zeros((height, width + 1))
walls_horizontal = np.zeros((height + 1, width))

# Define boundary walls
walls_vertical[:, [0, -1]] = 1
walls_horizontal[[0, -1], :] = 1

# Add internal walls
walls_vertical[1, 3] = 1
walls_horizontal[2, 2] = 1

# Create the maze
custom_maze = Maze(
    width=width,
    height=height,
    walls=[walls_vertical, walls_horizontal],
    start_position=(0, 0),
    end_position=(3, 5)
)
```

### Performance Analysis

```python
import matplotlib.pyplot as plt

# Collect scores during training
scores = []
for episode in range(1000):
    # ... training ...
    scores.append(env.step_numbers)

# Visualize evolution
plt.figure(figsize=(12, 5))
plt.plot(scores)
plt.title('Score Evolution per Episode')
plt.xlabel('Episode')
plt.ylabel('Number of Steps')
plt.show()
```

## 🔧 Development

### Project Structure

```
maze/
├── src/
│   ├── maze.py                 # Environment classes
│   ├── agent_epsilon_greedy.py # Learning agent
│   ├── config.py              # Configuration and utilities
│   ├── slider.py              # Slider widget for visualization
│   └── main.py                # Main demonstration script
├── setup.py                   # Installation configuration
├── README.md                  # Documentation
├── LICENSE                    # MIT License
└── requirements.txt           # Dependencies (optional)
```

### Development Installation

```bash
git clone https://github.com/V1centFaure/maze.git
cd maze
pip install -e .[dev]
```

### Testing and Code Quality

```bash
# Install development tools
pip install -e .[dev]

# Run tests
pytest

# Check code style
flake8 src/
black src/

# Type checking
mypy src/
```

## 📚 Usage Examples

### Q-learning vs SARSA Comparison

```python
# Train with Q-learning
agent_q = Agent_epsilon_greedy(env.height, env.width)
scores_q = train_agent(agent_q, env, method='q_learning')

# Train with SARSA
agent_sarsa = Agent_epsilon_greedy(env.height, env.width)
scores_sarsa = train_agent(agent_sarsa, env, method='sarsa')

# Compare performance
plt.plot(scores_q, label='Q-learning')
plt.plot(scores_sarsa, label='SARSA')
plt.legend()
plt.show()
```

### Convergence Analysis

```python
# Analyze convergence with different parameters
epsilons = [0.05, 0.1, 0.2, 0.3]
results = {}

for eps in epsilons:
    agent = Agent_epsilon_greedy(env.height, env.width, epsilon=eps)
    scores = train_agent(agent, env, episodes=500)
    results[eps] = scores

# Visualize results
for eps, scores in results.items():
    plt.plot(scores, label=f'ε={eps}')
plt.legend()
plt.title('Impact of Exploration Rate on Learning')
plt.show()
```

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the project
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Developed by [V1centFaure](https://github.com/V1centFaure)
- Inspired by classic reinforcement learning algorithms
- Uses Pygame for interactive visualization

## 📞 Support

To report bugs or request features:
- Open an [issue](https://github.com/V1centFaure/maze/issues) on GitHub
- Check the [documentation](https://github.com/V1centFaure/maze/blob/main/README.md)

---

**MAZE** - A modern approach to reinforcement learning in navigation environments 🎯
