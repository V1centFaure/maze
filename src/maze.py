"""
Maze visualization module using Pygame.

This module provides a Maze class for creating and visualizing mazes with walls
and optional value displays. It uses numpy for efficient array operations and
pygame for rendering.
"""

import numpy as np
import pygame

# Color constants
BLACK = (0, 0, 0)

# Display constants
MARGE = 20  # Margin around the maze display in pixels

class Maze:
    def __init__(self, width, height, walls=None, cell_size=100):
        """
        Initialize a new Maze instance.
        
        Args:
            width (int): Number of columns in the maze grid
            height (int): Number of rows in the maze grid
            walls (list, optional): List containing [vertical_walls, horizontal_walls] arrays.
                                  If None, creates boundary walls only.
            cell_size (int, optional): Size of each cell in pixels. Defaults to 100.
        
        Raises:
            ValueError: If walls dimensions don't match the expected grid dimensions.
        """
        # Store maze dimensions
        self.width = width
        self.height = height
        
        # Create the main grid (currently unused but kept for potential future use)
        self.grid = np.zeros(shape=(self.height, self.width), dtype=int)
        
        # Store cell size for rendering
        self.cell_size = cell_size
        
        # Initialize walls
        self.walls = walls
        if walls is None:
            # Create default walls (only boundary walls)
            # Note: HEIGHT and WIDTH are used from the main section - this may cause issues
            walls_vertical = np.zeros(shape=(HEIGHT, WIDTH+1))
            walls_horizontal = np.zeros(shape=(HEIGHT+1, WIDTH))
            
            # Set boundary walls
            walls_vertical[[0, -1], :] = 1  # Left and right boundaries
            walls_horizontal[:, [0, -1]] = 1  # Top and bottom boundaries
            
            self.walls = [walls_vertical, walls_horizontal]
            
        # Validate wall dimensions
        if not (self.walls[0].shape[0] == self.grid.shape[0] and 
                self.walls[0].shape[1] == self.grid.shape[1]+1 and
                self.walls[1].shape[0] == self.grid.shape[0]+1 and 
                self.walls[1].shape[1] == self.grid.shape[1]):
            raise ValueError(f"Wall matrix dimensions: expected [{self.grid.shape[0]}x{self.grid.shape[1]+1}, {self.grid.shape[0]+1}x{self.grid.shape[1]}], got [{self.walls[0].shape},{self.walls[1].shape}]")

    def draw(self, draw_value=False, q_values=None, grid=True):
        """
        Display the maze in a pygame window.
        
        Args:
            draw_value (bool, optional): Whether to display values in cells. Defaults to False.
            q_values (numpy.ndarray, optional): Array of values to display in cells.
                                              Must match maze dimensions if provided.
            grid (bool, optional): Whether to show grid lines. Defaults to True.
        """
        # Initialize Pygame
        pygame.init()

        # Create the window
        window_width = self.width * self.cell_size 
        window_height = self.height * self.cell_size
        
        window = pygame.display.set_mode((window_width + 2 * MARGE, window_height + 2 * MARGE))
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
        
        Args:
            window: Pygame surface to draw on
            draw_values (bool, optional): Whether to display values in cells. Defaults to False.
            q_values (numpy.ndarray, optional): Array of values to display in cells.
                                              Must match maze dimensions if provided.
            grid (bool, optional): Whether to show grid lines. Defaults to True.
        
        Raises:
            ValueError: If q_values dimensions don't match the maze grid dimensions.
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
                else:
                    line_width = default_line_width  # Thin or no line for grid
                    
                # Draw the vertical line
                pygame.draw.line(window, 
                                BLACK, 
                                (MARGE + w * self.cell_size, MARGE + h * self.cell_size), 
                                (MARGE + w * self.cell_size, MARGE + (h + 1) * self.cell_size), 
                                width=line_width)
                                
        # Draw horizontal lines (including walls)
        for h in range(self.height + 1):
            for w in range(self.width):
                # Check if this is a wall (thick line) or regular grid line
                if self.walls[1][h, w] == 1:
                    line_width = 10  # Thick line for walls
                else:
                    line_width = default_line_width  # Thin or no line for grid
                    
                # Draw the horizontal line
                pygame.draw.line(window, 
                                BLACK, 
                                (MARGE + w * self.cell_size, MARGE + h * self.cell_size), 
                                (MARGE + (w + 1) * self.cell_size, MARGE + h * self.cell_size), 
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
                    text_x = MARGE + (w + 0.5) * self.cell_size - text_size[0] // 2
                    text_y = MARGE + (h + 0.5) * self.cell_size - text_size[1] // 2
                    
                    window.blit(text, (text_x, text_y))
                
if __name__ == '__main__':
    # Example usage and testing
    WIDTH = 8
    HEIGHT = 4
    
    # Create wall arrays
    walls_vertical = np.zeros(shape=(HEIGHT, WIDTH+1))
    walls_horizontal = np.zeros(shape=(HEIGHT+1, WIDTH))
    
    # Set boundary walls
    walls_vertical[:, [0, -1]] = 1  # Left and right boundaries
    walls_horizontal[[0, -1], :] = 1  # Top and bottom boundaries

    # Add some internal walls for demonstration
    walls_vertical[[0, 1], 4] = 1  # Vertical wall at column 4, rows 0-1
    walls_vertical[[2, 3], 5] = 1  # Vertical wall at column 5, rows 2-3
    walls_vertical[2, 2] = 1       # Vertical wall at column 2, row 2
    walls_vertical[[1, 2], 6] = 1  # Vertical wall at column 6, rows 1-2

    # Add horizontal walls
    walls_horizontal[1, 2] = 1        # Horizontal wall at row 1, column 2
    walls_horizontal[3, [2, 3]] = 1   # Horizontal wall at row 3, columns 2-3
    walls_horizontal[1, 6] = 1        # Horizontal wall at row 1, column 6

    # Combine walls into the expected format
    walls = [walls_vertical, walls_horizontal]

    # Create maze instance
    maze = Maze(width=WIDTH, height=HEIGHT, walls=walls)
    
    # Create sample q_values for demonstration
    q_values = np.ones(shape=(HEIGHT, WIDTH))

    # Display the maze (without values, without grid)
    maze.draw(draw_value=False, q_values=q_values, grid=False)
