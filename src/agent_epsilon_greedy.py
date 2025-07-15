import config
import numpy as np

class Agent_epsilon_greedy:
    def __init__(self, height, width, epsilon=0.1, alpha=0.1, gamma=0.9):
        # Q-table : (états possibles, actions)
        # États : 0 à 21 + état "bust" (22) = 23 états
        # Actions : 0=stop, 1=tirer
        self.heigth = height
        self.width = width
        self.q_table = np.zeros(shape=(self.height, self.width))
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

    def select_action(self, state):
        #epsilon_greedy policy
        if np.random.random() < self.epsilon:  # Exploration
            return np.random.choice(['up', 'down', 'left', 'right'])
        else: #Exploitation
            match config.argmax(self.q_table[state, :]):
                case 0:
                    return 'up'
                case 1:
                    return 'down'
                case 2:
                    return 'left'
                case 3:
                    return 'right'

        