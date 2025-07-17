import pygame
import numpy as np
import ui
from maze import Maze
import config

class Create_maze:
    def __init__(self):
        self.button_loc = (10, 10)
        self.height = 0
        self.width = 0
        self.start_position=(0, 0),
        self.end_position = (1, 1)
        self.max_case = 20
        self.cell_size = 40
        self.wall_horizontal = np.zeros(shape=(self.max_case+1, self.max_case))
        self.wall_vertical = np.zeros(shape=(self.max_case, self.max_case+1))
        self.window_height = self.max_case * self.cell_size + 50 + 2*config.MARGE
        self.window_width = self.max_case * self.cell_size + 2*config.MARGE

    def create_maze(self):

        pygame.init()
        pygame.font.init()

        window = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("Maze")

        # Créer une police
        police = pygame.font.Font(None, 36)

        # Créer un objet texte
        texte_height = police.render("Height", True, (0, 0, 0))
        texte_width = police.render("Width", True, (0, 0, 0))




        stepper1 = ui.NumericStepper((self.button_loc[0]+100,self.button_loc[1]),
                                     (35,30), 
                                     min_val=3, 
                                     max_val=self.max_case, 
                                     initial=10, 
                                     step=1)
        stepper2 = ui.NumericStepper((self.button_loc[0]+300,self.button_loc[1]),
                                     (35,30), 
                                     min_val=3, 
                                     max_val=self.max_case, 
                                     initial=10, 
                                     step=1)
        
        
        clock = pygame.time.Clock()
        running = True
        while running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                stepper1.handle_event(event)
                stepper2.handle_event(event)           
            # Fill the window with white background
            window.fill((255, 255, 255))

            window.blit(texte_height, self.button_loc)
            window.blit(texte_width, (self.button_loc[0]+200, self.button_loc[1]))

            stepper1.draw(window)
            stepper2.draw(window)


            self.height = stepper1.get_value()
            self.width = stepper2.get_value()
            
            
            self.update_walls()
            self.draw(window)
            # Update the display
            pygame.display.flip()
            clock.tick(10)

        # Clean up and quit
        pygame.quit()

    def update_walls(self):
        pass

            
    def draw(self, window):
        # Draw vertical lines (including walls)
        for w in range(self.width + 1):
            for h in range(self.height):

                if self.wall_vertical[h, w] == 1:
                    line_width = 10  # Thick line for walls
                    color = config.WALL_COLOR
                else:

                    line_width = 1  # Thin or no line for grid
                    color = config.LINE_COLOR
                # Draw the vertical line
                pygame.draw.line(window, 
                                color, 
                                (config.MARGE + w * self.cell_size, 50 + h * self.cell_size), 
                                (config.MARGE + w * self.cell_size, 50 + (h + 1) * self.cell_size), 
                                width=line_width)
                                
        # Draw horizontal lines (including walls)
        for h in range(self.height + 1):
            for w in range(self.width):
                # Check if this is a wall (thick line) or regular grid line

                if self.wall_horizontal[h, w] == 1:
                    line_width = 10  # Thick line for walls
                    color = config.WALL_COLOR
                else:
                    line_width = 1  # Thin or no line for grid
                    color = config.LINE_COLOR
                # Draw the horizontal line
                pygame.draw.line(window, 
                                color, 
                                (config.MARGE + w * self.cell_size, 50 + h * self.cell_size), 
                                (config.MARGE + (w + 1) * self.cell_size, 50 + h * self.cell_size), 
                                width=line_width)


if __name__=='__main__':
    cm = Create_maze()
    cm.create_maze()