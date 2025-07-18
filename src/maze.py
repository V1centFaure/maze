"""
Maze visualization and simulation module using Pygame.

This module provides comprehensive maze functionality including:
- Maze creation with customizable walls and dimensions
- Interactive visualization using Pygame
- Agent movement simulation with collision detection
- Support for reinforcement learning environments
- Multiple predefined test mazes with varying complexity

The module uses numpy for efficient array operations and pygame for rendering.
It supports displaying Q-values or other numerical data within maze cells.

Classes:
    Maze: Main maze class with visualization and simulation capabilities
    Test_maze: Medium-sized test maze (8x5) for standard experiments
    Test_maze_little: Small test maze (4x3) for quick testing
    Test_maze_with_traps: Complex maze with variable rewards and traps

Example:
    >>> maze = Test_maze()
    >>> maze.draw()
    >>> next_state, reward, done = maze.step('right')
"""
import config
import numpy as np
import pygame



class Maze:
    def __init__(self, 
                 width, 
                 height, 
                 walls=None,
                 grid=None, 
                 cell_size=100, 
                 start_position=(0, 0),
                 end_position=(1, 1)):
        """
        Initialize a new Maze instance.
        
        Creates a maze with specified dimensions, walls, and reward structure.
        The maze uses a coordinate system where (0,0) is the top-left corner.
        
        Args:
            width (int): Number of columns in the maze grid. Must be positive.
            height (int): Number of rows in the maze grid. Must be positive.
            walls (list, optional): List containing [vertical_walls, horizontal_walls] arrays.
                                  vertical_walls: shape (height, width+1)
                                  horizontal_walls: shape (height+1, width)
                                  If None, creates boundary walls only.
            grid (numpy.ndarray, optional): Reward grid with shape (height, width).
                                          If None, creates default grid with time penalties
                                          and goal reward.
            cell_size (int, optional): Size of each cell in pixels for visualization.
                                     Defaults to 100.
            start_position (tuple, optional): Agent's starting position as (row, col).
                                            Defaults to (0, 0).
            end_position (tuple, optional): Goal position as (row, col).
                                          Defaults to (1, 1).
        
        Attributes:
            agent_pos (tuple): Current agent position
            step_numbers (int): Number of steps taken by the agent
            
        Raises:
            ValueError: If walls dimensions don't match the expected grid dimensions.
            
        Note:
            Wall arrays use a specific indexing system:
            - vertical_walls[row, col] represents the wall to the left of cell (row, col)
            - horizontal_walls[row, col] represents the wall above cell (row, col)
        """
            # Store maze dimensions
        self.width = width
        self.height = height
        self.walls = walls
        self.cell_size = cell_size
        self.start_position = start_position
        self.end_position = end_position
        self.grid = grid

        # Create the main grid (currently unused but kept for potential future use)
        if self.grid is None:
            self.grid = np.ones(shape=(self.height, self.width), dtype=int) * config.TIME_PENALTY
            self.grid[self.end_position[0], self.end_position[1]] = config.GOAL_REWARD

        if walls is None:
            # Create default walls (only boundary walls)
            # Note: HEIGHT and WIDTH are used from the main section - this may cause issues
            walls_vertical = np.zeros(shape=(self.height, self.width+1))
            walls_horizontal = np.zeros(shape=(self.height+1, self.width))
            
            # Set boundary walls
            walls_vertical[[0, -1], :] = 1  # Left and right boundaries
            walls_horizontal[:, [0, -1]] = 1  # Top and bottom boundaries
            
            self.walls = [walls_vertical, walls_horizontal]

        self.agent_pos = self.start_position
        self.step_numbers = 0
            
        # Validate wall dimensions
        if not (self.walls[0].shape[0] == self.grid.shape[0] and 
                self.walls[0].shape[1] == self.grid.shape[1]+1 and
                self.walls[1].shape[0] == self.grid.shape[0]+1 and 
                self.walls[1].shape[1] == self.grid.shape[1]):
            raise ValueError(f"Wall matrix dimensions: expected [{self.grid.shape[0]}x{self.grid.shape[1]+1}, {self.grid.shape[0]+1}x{self.grid.shape[1]}], got [{self.walls[0].shape},{self.walls[1].shape}]")


    def draw(self, draw_value=False, q_values=None, grid=True):
        """
        Display the maze in an interactive pygame window.
        
        Opens a pygame window showing the maze with walls, optional grid lines,
        and optional numerical values in cells. The window remains open until
        the user closes it, allowing for interactive visualization.
        
        Args:
            draw_value (bool, optional): Whether to display numerical values in cells.
                                       Requires q_values to be provided. Defaults to False.
            q_values (numpy.ndarray, optional): Array of values to display in cells.
                                              Must have shape (height, width) matching
                                              the maze dimensions. Commonly used for
                                              Q-values in reinforcement learning.
            grid (bool, optional): Whether to show thin grid lines between cells.
                                 If False, only walls are visible. Defaults to True.
                                 
        Note:
            This method blocks execution until the pygame window is closed.
            The window size is automatically calculated based on maze dimensions
            and cell size, plus margins defined in the config module.
            
        Example:
            >>> maze = Test_maze()
            >>> q_vals = np.random.rand(maze.height, maze.width)
            >>> maze.draw(draw_value=True, q_values=q_vals, grid=False)
        """
        # Initialize Pygame
        pygame.init()

        # Create the window
        window_width = self.width * self.cell_size 
        window_height = self.height * self.cell_size
        
        window = pygame.display.set_mode((window_width + 2 * config.MARGIN, window_height + 2 * config.MARGIN))
        pygame.display.set_caption("Maze")

        # Main game loop
        running = True
        while running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # Fill the window with white background
            window.fill((255, 255, 255))

            # Draw the maze grid
            self.draw_grid(window, draw_values=draw_value, q_values=q_values, grid=grid)

            # Update the display
            pygame.display.flip()

        # Clean up and quit
        pygame.quit()
    
    def draw_grid(self, window, draw_values=False, q_values=None, grid=True):
        """
        Draw the maze grid with walls and optional values on a pygame surface.
        
        This method handles the actual rendering of the maze components:
        - Grid lines (if enabled)
        - Walls (thick lines)
        - Numerical values in cells (if provided)
        
        The rendering uses colors and dimensions defined in the config module.
        Walls are drawn as thick lines (width=10) while grid lines are thin (width=1).
        
        Args:
            window (pygame.Surface): Pygame surface to draw on. Must be large enough
                                   to accommodate the maze with margins.
            draw_values (bool, optional): Whether to display numerical values in cells.
                                        Requires q_values parameter. Defaults to False.
            q_values (numpy.ndarray, optional): Array of values to display in cells.
                                              Must have shape (height, width) matching
                                              maze dimensions. Values are rendered as
                                              text centered in each cell.
            grid (bool, optional): Whether to show thin grid lines between cells.
                                 If False, only walls are visible. Defaults to True.
        
        Raises:
            ValueError: If q_values dimensions don't match the maze grid dimensions
                       (expected shape: height × width).
                       
        Note:
            This method is typically called by the draw() method and not directly
            by users. It assumes pygame has been initialized and a valid surface
            is provided.
        """
        # Draw the grid lines and walls
        # Set default line width based on grid visibility
        if grid: 
            default_line_width = 1  # Show thin grid lines
        else:
            default_line_width = 0  # Hide grid lines
            
        # Draw vertical lines (including walls)
        for w in range(self.width + 1):
            for h in range(self.height):
                # Check if this is a wall (thick line) or regular grid line
                if self.walls[0][h, w] == 1:
                    line_width = 10  # Thick line for walls
                    color = config.WALL_COLOR
                else:
                    line_width = default_line_width  # Thin or no line for grid
                    color = config.LINE_COLOR
                # Draw the vertical line
                pygame.draw.line(window, 
                                color, 
                                (config.MARGIN + w * self.cell_size, config.MARGIN + h * self.cell_size), 
                                (config.MARGIN + w * self.cell_size, config.MARGIN + (h + 1) * self.cell_size), 
                                width=line_width)
                                
        # Draw horizontal lines (including walls)
        for h in range(self.height + 1):
            for w in range(self.width):
                # Check if this is a wall (thick line) or regular grid line
                if self.walls[1][h, w] == 1:
                    line_width = 10  # Thick line for walls
                    color = config.WALL_COLOR
                else:
                    line_width = default_line_width  # Thin or no line for grid
                    color = config.LINE_COLOR
                # Draw the horizontal line
                pygame.draw.line(window, 
                                color, 
                                (config.MARGIN + w * self.cell_size, config.MARGIN + h * self.cell_size), 
                                (config.MARGIN + (w + 1) * self.cell_size, config.MARGIN + h * self.cell_size), 
                                width=line_width)
                
        # Display values in cells if requested
        font = pygame.font.SysFont('Arial', 20)

        if draw_values and q_values is not None:
            # Validate q_values dimensions
            if not q_values.shape == self.grid.shape:
                raise ValueError(f"Incorrect matrix dimensions: expected {self.width}x{self.height}, got {q_values.shape}")

            # Render values in each cell
            for h in range(self.height):
                for w in range(self.width):
                    # Render the text
                    text = font.render(str(q_values[h][w]), 1, (0, 0, 0))
                    text_size = font.size(str(q_values[h][w]))
                    
                    # Center the text in the cell
                    text_x = config.MARGIN + (w + 0.5) * self.cell_size - text_size[0] // 2
                    text_y = config.MARGIN + (h + 0.5) * self.cell_size - text_size[1] // 2
                    
                    window.blit(text, (text_x, text_y))
    def reset(self):
        """
        Reset the maze to its initial state.
        
        Reinitializes the maze with the same parameters, effectively:
        - Resetting agent position to start_position
        - Resetting step counter to 0
        - Restoring original grid values
        
        This method is useful for running multiple episodes in reinforcement
        learning experiments without creating new maze instances.
        
        Note:
            This implementation recreates the entire maze object, which may
            not be the most efficient approach for frequent resets.
        """
        self.__init__(width=self.width,
                      height=self.height,
                      walls=self.walls,
                      cell_size=self.cell_size,
                      start_position=self.start_position,
                      end_position=self.end_position)
        
    def step(self, action):
        """
        Execute an action in the maze and return the result.
        
        This method simulates one step of agent movement in the maze environment.
        It handles collision detection with walls, position updates, reward
        calculation, and terminal state detection.
        
        The coordinate system uses (row, col) indexing where:
        - 'up' decreases row index
        - 'down' increases row index  
        - 'left' decreases column index
        - 'right' increases column index
        
        Args:
            action (str): Action to execute. Must be one of:
                         'up', 'down', 'left', 'right'
            
        Returns:
            tuple: (next_state, reward, done) where:
                - next_state (tuple or None): New agent position as (row, col)
                                             or None if move is blocked by wall
                - reward (float or None): Reward obtained from the new cell
                                        or None if move is impossible
                - done (bool): True if goal position is reached, False otherwise
                
        Side Effects:
            - Updates self.agent_pos if move is valid
            - Increments self.step_numbers counter
            
        Note:
            Wall collision detection checks the appropriate wall array:
            - Vertical movements check horizontal_walls
            - Horizontal movements check vertical_walls
            
        Example:
            >>> maze = Test_maze()
            >>> next_state, reward, done = maze.step('right')
            >>> if next_state is not None:
            ...     print(f"Moved to {next_state}, got reward {reward}")
        """
        done = False
        
        # Determine coordinates based on action
        match action:
            case 'up':
                # To go up, check horizontal wall above
                coord_check = (self.agent_pos[0], self.agent_pos[1])
                new_coord = (self.agent_pos[0] - 1, self.agent_pos[1])
            case 'down':
                # To go down, check horizontal wall below
                coord_check = (self.agent_pos[0] + 1, self.agent_pos[1])
                new_coord = (self.agent_pos[0] + 1, self.agent_pos[1])               
            case 'left':
                # To go left, check vertical wall to the left
                coord_check = (self.agent_pos[0], self.agent_pos[1])
                new_coord = (self.agent_pos[0], self.agent_pos[1] - 1)
            case 'right':
                # To go right, check vertical wall to the right
                coord_check = (self.agent_pos[0], self.agent_pos[1] + 1)
                new_coord = (self.agent_pos[0], self.agent_pos[1] + 1)
        
        # Select wall type to check (0=vertical, 1=horizontal)
        wall_type = 1 if action in ['up', 'down'] else 0

        # Check for wall collision
        if self.walls[wall_type][coord_check] == 1:
            # Impossible move: wall collision
            return None, None, done
        else:
            # Valid move: update position and calculate reward
            self.step_numbers += 1
            self.agent_pos = new_coord
            next_state = self.agent_pos
            
            # Check if goal is reached
            if self.grid[self.agent_pos] == config.GOAL_REWARD:
                reward = self.grid[self.agent_pos]
                done = True
            else:
                reward = self.grid[self.agent_pos]
                
        return next_state, reward, done



class Test_maze(Maze):
    """
    Medium-sized test maze (8x5) with some internal walls.
    
    This maze serves as a standard test case for training reinforcement
    learning agents. It contains strategically placed internal walls
    to create interesting navigation challenges.
    """
    
    def __init__(self):
        """
        Initialize the test maze with a predefined configuration.
        
        Configuration:
        - Dimensions: 8 columns × 5 rows
        - Start position: (0, 0) - top-left corner
        - End position: (4, 7) - bottom-right corner
        - Internal walls arranged to create alternative paths
        """
        # Maze parameters
        self.width = 8
        self.height = 5
        self.cell_size = 100
        self.start_position = (0, 0)
        self.end_position = (4, 7)
        
        # Create wall matrices
        walls_vertical = np.zeros(shape=(self.height, self.width + 1))
        walls_horizontal = np.zeros(shape=(self.height + 1, self.width))
        
        # Define boundary walls
        walls_vertical[:, [0, -1]] = 1  # Left and right walls
        walls_horizontal[[0, -1], :] = 1  # Top and bottom walls

        # Add internal walls to create obstacles
        walls_vertical[[0, 1], 4] = 1  # Vertical wall at column 4, rows 0-1
        walls_vertical[[2, 3], 5] = 1  # Vertical wall at column 5, rows 2-3
        walls_vertical[2, 2] = 1       # Vertical wall at column 2, row 2
        walls_vertical[[1, 2], 6] = 1  # Vertical wall at column 6, rows 1-2
        walls_vertical[4, 3] = 1       # Vertical wall at column 3, row 4

        # Add horizontal walls
        walls_horizontal[1, 2] = 1        # Horizontal wall at row 1, column 2
        walls_horizontal[3, [2, 3]] = 1   # Horizontal wall at row 3, columns 2-3
        walls_horizontal[1, 6] = 1        # Horizontal wall at row 1, column 6

        # Assemble walls in expected format
        self.walls = [walls_vertical, walls_horizontal]

        # Initialize parent class
        super().__init__(
            width=self.width, 
            height=self.height, 
            walls=self.walls, 
            cell_size=self.cell_size,
            start_position=self.start_position,
            end_position=self.end_position
        )
        
    def reset(self):
        """Reset the maze to its initial state."""
        self.__init__()


class Test_maze_little(Maze):
    """
    Small test maze (4x3) for quick experiments.
    
    This miniature maze is ideal for quickly testing algorithms
    or for demonstrations, thanks to its reduced size that allows
    rapid learning convergence.
    """
    
    def __init__(self):
        """
        Initialize the small test maze.
        
        Configuration:
        - Dimensions: 4 columns × 3 rows
        - Start position: (0, 0) - top-left corner
        - End position: (2, 3) - bottom-right corner
        - Few internal walls to create minimal challenge
        """
        # Maze parameters
        self.width = 4
        self.height = 3
        self.cell_size = 100
        self.start_position = (0, 0)
        self.end_position = (2, 3)

        # Create wall matrices
        walls_vertical = np.zeros(shape=(self.height, self.width + 1))
        walls_horizontal = np.zeros(shape=(self.height + 1, self.width))
        
        # Define boundary walls
        walls_vertical[:, [0, -1]] = 1  # Left and right walls
        walls_horizontal[[0, -1], :] = 1  # Top and bottom walls

        # Add some strategic internal walls
        walls_vertical[0, 1] = 1  # Vertical wall at column 1, row 0
        walls_vertical[1, 2] = 1  # Vertical wall at column 2, row 1
        walls_vertical[2, 3] = 1  # Vertical wall at column 3, row 2

        # Add one horizontal wall
        walls_horizontal[1, 2] = 1  # Horizontal wall at row 1, column 2

        # Assemble walls in expected format
        self.walls = [walls_vertical, walls_horizontal]
        
        # Initialize parent class
        super().__init__(
            width=self.width, 
            height=self.height, 
            walls=self.walls, 
            cell_size=self.cell_size,
            start_position=self.start_position,
            end_position=self.end_position
        )
        
    def reset(self):
        """Reset the maze to its initial state."""
        self.__init__()


class Test_maze_with_traps(Maze):
    """
    Complex maze with traps and variable rewards.
    
    This advanced maze features different types of cells with varied
    rewards, including traps (very negative rewards) that make navigation
    more complex and realistic. It's ideal for testing reinforcement learning
    algorithms in a more sophisticated environment.
    """
    
    def __init__(self):
        """
        Initialize the maze with traps.
        
        Configuration:
        - Dimensions: 7 columns × 8 rows
        - Start position: (2, 3) - center-left
        - End position: (5, 3) - center, lower
        - Custom reward grid with:
          * Normal cells: -1 (time penalty)
          * Light traps: -5 (moderate penalty)
          * Heavy traps: -10, -30 (severe penalties)
          * Goal: +100 (final reward)
        """
        # Maze parameters
        self.width = 7
        self.height = 8
        self.cell_size = 100
        self.start_position = (2, 3)
        self.end_position = (5, 3)
        
        # Custom reward grid with different cell types
        self.grid = np.array([
            [-1, -1, -1, -1, -1, -1, -1],      # Row 0: normal cells
            [-1, -1, -1, -1, -1, -1, -1],      # Row 1: normal cells
            [-1, -1, -1, -1, -1, -1, -1],      # Row 2: normal cells (start position)
            [-1, -1, -1, -30, -5, -1, -1],     # Row 3: heavy and light traps
            [-1, -1, -5, -30, -1, -5, -1],     # Row 4: mix of traps
            [-1, -1, -30, config.GOAL_REWARD, -5, -1, -1],  # Row 5: goal surrounded by traps
            [-1, -1, -1, -1, -1, -10, -1],     # Row 6: medium trap
            [-1, -1, -1, -1, -1, -1, -1]       # Row 7: normal cells
        ])
        
        # Horizontal walls configuration (between rows)
        walls_horizontal = np.array([
            [1, 1, 1, 1, 1, 1, 1],  # Top wall (boundary)
            [0, 0, 0, 0, 1, 0, 0],  # Some openings
            [0, 1, 1, 0, 0, 0, 0],  # Internal walls
            [0, 0, 0, 0, 0, 0, 0],  # Free passage
            [0, 1, 0, 0, 0, 1, 0],  # Strategic walls
            [0, 0, 1, 0, 1, 0, 0],  # Around goal
            [0, 1, 0, 0, 1, 0, 0],  # Internal walls
            [0, 0, 0, 0, 0, 0, 0],  # Free passage
            [1, 1, 1, 1, 1, 1, 1]   # Bottom wall (boundary)
        ])
        
        # Vertical walls configuration (between columns)
        walls_vertical = np.array([
            [1, 0, 0, 1, 0, 0, 0, 1],  # Boundaries and some internal walls
            [1, 0, 0, 0, 1, 1, 0, 1],  # Walls creating corridors
            [1, 1, 0, 0, 0, 1, 0, 1],  # Around start position
            [1, 0, 0, 1, 1, 1, 0, 1],  # Complex walls
            [1, 0, 0, 1, 1, 0, 1, 1],  # Around traps
            [1, 0, 1, 0, 0, 0, 1, 1],  # Around goal
            [1, 0, 0, 1, 0, 0, 0, 1],  # Internal walls
            [1, 0, 0, 0, 0, 0, 0, 1],  # Boundaries
        ])
        
        # Assemble walls in expected format
        self.walls = [walls_vertical, walls_horizontal]

        # Initialize parent class with custom grid
        super().__init__(
            width=self.width, 
            height=self.height, 
            walls=self.walls, 
            cell_size=self.cell_size,
            grid=self.grid,
            start_position=self.start_position,
            end_position=self.end_position
        )
    
    def reset(self):
        """Reset the maze to its initial state."""
        self.__init__()
        

if __name__ == '__main__':

    pass





    
    



    # # Create sample q_values for demonstration
    # q_values = np.ones(shape=(env.height, env.width))

    # # Display the maze (without values, without grid)
    # env.draw(draw_value=True, q_values=q_values, grid=False)
