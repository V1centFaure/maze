import numpy as np

# Color constants
BLACK = (0, 0, 0)

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
