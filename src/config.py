import numpy as np
import pygame
import pickle
from slider import Slider

# Color constants
LINE_COLOR = (200, 200, 200)
WALL_COLOR = (100, 50, 50)
PATH_COLOR = (255, 100, 100)

# Display constants
MARGE = 20  # Margin around the maze display in pixels

TIME_PENALTY = -1
GOAL_REWARD = 100

ACTION2IDX = {'up': 0, 'down': 1, 'left': 2, 'right': 3}
IDX2ACTION = {0 :'up', 1: 'down', 2: 'left', 3: 'right'}

def argmax(q_values: list) -> int:
    """
    Takes in a list of q_values and returns the index of the item
    with the highest value. Breaks ties randomly.
    returns: int - the index of the highest value in q_values
    """
    arr_q_values = np.array(q_values)
    return np.random.choice(np.where(arr_q_values == arr_q_values.max())[0])

def draw_maze_with_path(env, paths):
    pygame.init()
    # Create the window
    window_width = env.width * env.cell_size 
    window_height = env.height * env.cell_size
    
    window = pygame.display.set_mode((window_width + 2 * MARGE, window_height + 2 * MARGE))
    pygame.display.set_caption("Maze")
    slider = Slider((0 + MARGE, window_height + 1.5*MARGE), (window_width, MARGE//3), 0, len(paths)-1, 0)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            slider.handle_event(event)


        # Fill the window with white background
        window.fill((255, 255, 255))

        window.fill(color = (100, 255, 100),
            rect=pygame.Rect(env.start_position[1]*env.cell_size + MARGE,
                                env.start_position[0]*env.cell_size + MARGE,
                                env.cell_size,
                                env.cell_size
                                )
                    )
        window.fill(color = (0, 0, 150),
            rect=pygame.Rect(env.end_position[1]*env.cell_size + MARGE,
                                env.end_position[0]*env.cell_size + MARGE,
                                env.cell_size,
                                env.cell_size
                                )
                    )


        # Draw the maze grid
        env.draw_grid(window)

        episode_idx = slider.get_value()
        path = paths[episode_idx]

        #Dessiner path
        pygame.draw.circle(window, 
                           (0, 150, 0), 
                           (env.start_position[0] * env.cell_size+ MARGE + env.cell_size//2, 
                            env.start_position[1] * env.cell_size+ MARGE + env.cell_size//2),
                             10)
        for p in range(len(path)-1):
            pygame.draw.line(window, 
                             PATH_COLOR, 
                             (path[p][1] * env.cell_size+ MARGE + env.cell_size//2, path[p][0] * env.cell_size+ MARGE + env.cell_size//2), 
                             (path[p+1][1] * env.cell_size+ MARGE + env.cell_size//2, path[p+1][0] * env.cell_size+ MARGE + env.cell_size//2),
                             2)




 

        slider.draw(window)
        
 
                    

        # Update the display
        pygame.display.flip()
    pygame.quit()





if __name__ == "__main__":
    from maze import Test_maze_little, Test_maze
    env = Test_maze()
    print(env.start_position, env.end_position)

    with open("paths.pkl", "rb") as f:
        paths = pickle.load(f)
        print(paths)
    draw_maze_with_path(env, paths)