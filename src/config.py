"""
Configuration module and utilities for maze visualization and reinforcement learning.

This module provides essential configuration constants, utility functions, and
visualization capabilities for the maze reinforcement learning environment. It serves
as a centralized location for all configurable parameters and shared functionality
across the maze learning system.

Key Components:
- Color constants for consistent visual styling
- Display parameters for maze rendering
- Reward structure configuration for reinforcement learning
- Action mapping utilities for agent-environment interaction
- Utility functions for Q-value processing
- Advanced path visualization with interactive controls

The module supports both basic maze display and sophisticated path visualization
with episode navigation capabilities, making it suitable for both training
and analysis of reinforcement learning agents.

Constants:
    LINE_COLOR: Grid line color for maze visualization
    WALL_COLOR: Wall color for maze boundaries and obstacles
    PATH_COLOR: Color for displaying agent paths
    MARGIN: Display margin around maze
    TIME_PENALTY: Step penalty for reinforcement learning
    GOAL_REWARD: Reward for reaching the goal state
    ACTION2IDX: Mapping from action strings to indices
    IDX2ACTION: Mapping from indices to action strings

Functions:
    argmax: Robust maximum value index finder with tie-breaking
    draw_maze_with_path: Interactive path visualization with episode navigation

Example:
    >>> import config
    >>> action_idx = config.ACTION2IDX['right']  # Get action index
    >>> best_action_idx = config.argmax([0.1, 0.3, 0.2, 0.3])  # Find best action
    >>> config.draw_maze_with_path(maze, episode_paths)  # Visualize training results
"""

import numpy as np
import pygame
import pickle
from slider import Slider

# ============================================================================
# COLOR CONSTANTS
# ============================================================================
"""
Color definitions for consistent maze visualization.

These RGB color tuples define the visual appearance of different maze elements.
All colors are specified as (Red, Green, Blue) tuples with values 0-255.
"""
LINE_COLOR = (200, 200, 200)  # Grid line color (light gray) - subtle grid lines
WALL_COLOR = (100, 50, 50)    # Wall color (dark brown) - prominent obstacles  
PATH_COLOR = (255, 100, 100)  # Path color (light red) - agent trajectory

# ============================================================================
# DISPLAY CONSTANTS
# ============================================================================
"""
Display parameters for maze rendering and window layout.

These constants control the visual layout and spacing of the maze display,
ensuring consistent appearance across different maze sizes.
"""
MARGIN = 20  # Margin around maze display in pixels - provides visual breathing room

# ============================================================================
# REWARD CONSTANTS
# ============================================================================
"""
Reward structure configuration for reinforcement learning.

These values define the reward signal that guides agent learning:
- TIME_PENALTY: Encourages efficient navigation by penalizing each step
- GOAL_REWARD: Provides strong positive reinforcement for task completion

The reward structure creates a trade-off between speed and goal achievement,
encouraging the agent to find the shortest path to the goal.
"""
TIME_PENALTY = -1   # Penalty for each time step - encourages efficiency
GOAL_REWARD = 100   # Reward for reaching the goal - strong positive signal

# ============================================================================
# ACTION MAPPINGS
# ============================================================================
"""
Bidirectional mapping between action representations.

These dictionaries provide conversion between human-readable action strings
and numerical indices used internally by the learning algorithms. The mapping
ensures consistency across the entire system.

Action encoding:
- 0/'up': Move up (decrease row index)
- 1/'down': Move down (increase row index)  
- 2/'left': Move left (decrease column index)
- 3/'right': Move right (increase column index)
"""
ACTION2IDX = {'up': 0, 'down': 1, 'left': 2, 'right': 3}  # String to index mapping
IDX2ACTION = {0: 'up', 1: 'down', 2: 'left', 3: 'right'}  # Index to string mapping


def argmax(q_values):
    """
    Return the index of the maximum value with random tie-breaking.
    
    This function implements a robust argmax operation that handles ties
    by randomly selecting among all indices that achieve the maximum value.
    This is crucial in reinforcement learning to avoid systematic bias
    toward lower-indexed actions when Q-values are equal.
    
    The function is particularly important during the early stages of learning
    when many Q-values are identical (often zero), and during convergence
    when multiple actions may have similar optimal values.
    
    Args:
        q_values (array-like): Array of Q-values for different actions.
                             Can be any array-like structure (list, numpy array, etc.)
                             containing numerical values.
                             
    Returns:
        int: Index of the maximum value. If multiple indices have the same
             maximum value, one is chosen uniformly at random.
             
    Raises:
        ValueError: If q_values is empty or contains non-numerical values.
        
    Note:
        This function uses numpy's random number generator, so results
        depend on the current random state. For reproducible results,
        set the numpy random seed before calling this function.
        
    Example:
        >>> q_vals = [0.1, 0.3, 0.3, 0.2]  # Two actions tied for maximum
        >>> best_action = argmax(q_vals)  # Returns 1 or 2 randomly
        >>> 
        >>> # With all equal values (common in early learning)
        >>> q_vals = [0.0, 0.0, 0.0, 0.0]
        >>> best_action = argmax(q_vals)  # Returns 0, 1, 2, or 3 randomly
    """
    arr_q_values = np.array(q_values)
    return np.random.choice(np.where(arr_q_values == arr_q_values.max())[0])


def draw_maze_with_path(env, paths, grid=False):
    """
    Display an interactive maze visualization with episode path navigation.
    
    This function creates a sophisticated graphical interface for analyzing
    reinforcement learning training results. It displays the maze environment
    with an interactive slider that allows users to navigate through different
    training episodes and observe how the agent's behavior evolved over time.
    
    Features:
    - Interactive episode selection via slider control
    - Visual path tracing showing agent movement
    - Color-coded start (green) and goal (blue) positions
    - Optional reward value display in each cell
    - Real-time path visualization with smooth line rendering
    
    The visualization is particularly useful for:
    - Analyzing learning progress across episodes
    - Identifying convergence patterns
    - Debugging agent behavior
    - Demonstrating training results
    
    Args:
        env (Maze): Maze environment instance containing the maze structure,
                   dimensions, walls, and reward grid. Must have attributes:
                   width, height, cell_size, start_position, end_position,
                   grid, and draw_grid method.
        paths (list): List of episode paths, where each path is a list of
                     (row, col) tuples representing the agent's trajectory
                     through the maze. Each path should start at start_position
                     and typically end at end_position or when episode terminates.
        grid (bool, optional): Whether to display numerical reward values
                             within each maze cell. Useful for understanding
                             the reward structure. Defaults to False.
                             
    Raises:
        pygame.error: If pygame initialization fails or display cannot be created.
        AttributeError: If env lacks required attributes or methods.
        IndexError: If paths contain invalid coordinates for the given maze.
        
    Side Effects:
        - Opens a pygame window that blocks execution until closed
        - Initializes pygame display system
        - Creates interactive GUI elements
        - Handles user input events
        
    Controls:
        - Mouse: Click and drag slider to navigate between episodes
        - Window close button: Exit visualization
        
    Note:
        This function blocks execution until the user closes the window.
        The pygame window is automatically cleaned up when the function exits.
        
    Example:
        >>> from maze import Test_maze
        >>> import pickle
        >>> 
        >>> # Load maze and training results
        >>> maze = Test_maze()
        >>> with open('training_paths.pkl', 'rb') as f:
        ...     episode_paths = pickle.load(f)
        >>> 
        >>> # Launch interactive visualization
        >>> draw_maze_with_path(maze, episode_paths, grid=True)
    """
    # Initialize Pygame
    pygame.init()
    
    # Create display window
    window_width = env.width * env.cell_size 
    window_height = env.height * env.cell_size
    
    window = pygame.display.set_mode((window_width + 2 * MARGIN, window_height + 2 * MARGIN))
    pygame.display.set_caption("Maze - Path Visualization")
    
    # Create slider to navigate between episodes
    slider = Slider(
        (0 + MARGIN, window_height + 1.5 * MARGIN), 
        (window_width, MARGIN // 3), 
        0, len(paths) - 1, 0
    )

    # Main display loop
    running = True
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            slider.handle_event(event)

        # Fill background with white
        window.fill((255, 255, 255))

        # Display start position (light green)
        window.fill(
            color=(100, 255, 100),
            rect=pygame.Rect(
                env.start_position[1] * env.cell_size + MARGIN,
                env.start_position[0] * env.cell_size + MARGIN,
                env.cell_size,
                env.cell_size
            )
        )
        
        # Display end position (dark blue)
        window.fill(
            color=(0, 0, 150),
            rect=pygame.Rect(
                env.end_position[1] * env.cell_size + MARGIN,
                env.end_position[0] * env.cell_size + MARGIN,
                env.cell_size,
                env.cell_size
            )
        )
        
        # Optional display of reward values in each cell
        if grid:
            font = pygame.font.SysFont('Arial', 20)
            for h in range(env.height):
                for w in range(env.width):
                    # Render text with reward value
                    text = font.render(str(env.grid[h][w]), 1, (0, 0, 0))
                    text_size = font.size(str(env.grid[h][w]))

                    # Center text in cell
                    text_x = MARGIN + (w + 0.5) * env.cell_size - text_size[0] // 2
                    text_y = MARGIN + (h + 0.5) * env.cell_size - text_size[1] // 2
                    
                    window.blit(text, (text_x, text_y))
        
        # Display maze grid (walls and lines)
        env.draw_grid(window)

        # Get selected episode via slider
        episode_idx = slider.get_value()
        path = paths[episode_idx]

        # Display start point with green circle
        pygame.draw.circle(
            window, 
            (0, 150, 0), 
            (env.start_position[1] * env.cell_size + MARGIN + env.cell_size // 2, 
             env.start_position[0] * env.cell_size + MARGIN + env.cell_size // 2),
            10
        )
        
        # Display path taken with connected lines
        for p in range(len(path) - 1):
            pygame.draw.line(
                window, 
                PATH_COLOR, 
                (path[p][1] * env.cell_size + MARGIN + env.cell_size // 2, 
                 path[p][0] * env.cell_size + MARGIN + env.cell_size // 2), 
                (path[p+1][1] * env.cell_size + MARGIN + env.cell_size // 2, 
                 path[p+1][0] * env.cell_size + MARGIN + env.cell_size // 2),
                2
            )

        # Display slider
        slider.draw(window)

        # Update display
        pygame.display.flip()
    
    # Cleanup and close
    pygame.quit()


if __name__ == "__main__":
    """
    Demonstration and testing code for path visualization functionality.
    
    This section provides a complete example of how to use the path visualization
    system. It loads a predefined test maze and attempts to load saved training
    paths for interactive visualization.
    
    The demo showcases:
    - Maze environment creation
    - Path data loading from pickle files
    - Interactive visualization launch
    - Error handling for missing data files
    
    To use this demo:
    1. Run main.py first to generate training data and save paths
    2. Run this module directly to visualize the results
    
    Expected files:
    - paths.pkl: Pickle file containing list of episode paths
    
    Note:
        This demo requires that training has been performed and paths
        have been saved to 'paths.pkl'. If the file doesn't exist,
        an informative error message is displayed.
    """
    from maze import Test_maze_little, Test_maze
    
    # Create test maze
    env = Test_maze()
    print(f"Start position: {env.start_position}, End position: {env.end_position}")

    # Load saved paths
    try:
        with open("paths.pkl", "rb") as f:
            paths = pickle.load(f)
            print(f"Number of paths loaded: {len(paths)}")
        
        # Launch visualization
        draw_maze_with_path(env, paths)
    except FileNotFoundError:
        print("File paths.pkl not found. Run main.py first to generate paths.")
