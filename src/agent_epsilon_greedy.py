import config
import numpy as np
import datetime
import pickle


class Agent_epsilon_greedy:
    def __init__(self, height, width, epsilon=0.1, alpha=0.1, gamma=0.9):
        # Q-table : (états possibles, actions)
        # États : 0 à 21 + état "bust" (22) = 23 états
        # Actions : 0=stop, 1=tirer
        self.height = height
        self.width = width
        self.q_table = np.zeros(shape=(self.height, self.width, 4))
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

    def select_action(self, state):
        #epsilon_greedy policy
        if np.random.random() < self.epsilon:  # Exploration
            return np.random.choice(['up', 'down', 'left', 'right'])
        else: #Exploitation
            return config.IDX2ACTION[config.argmax(self.q_table[state[0], state[1], :])]
 

    def update(self, state, action, reward, next_state, done, method = "q_learning"):
        action_idx = config.ACTION2IDX[action]
        if method =='sarsa': 
            next_action = self.select_action(state=next_state)
            next_action_idx = config.ACTION2IDX[next_action]
        if done:
            target = reward
        else:
            if method == 'sarsa':
                target = reward + self.gamma * self.q_table[next_state[0], next_state[1], next_action_idx]
            elif method == 'q_learning':
                target = reward + self.gamma * np.max(self.q_table[next_state[0], next_state[1], :])
        
        self.q_table[state[0], state[1], action_idx] += self.alpha * (target - self.q_table[state[0], state[1], action_idx])

    def reset(self, height, width, epsilon, alpha, gamma):
        if height is None: height = self.height
        if width is None: width = self.width
        if epsilon is None: epsilon = self.epsilon
        if alpha is None: alpha = self.alpha
        if gamma is None: gamma = self.gamma
        
        self.__init__(height = height,
                      width = width,
                      epsilon = epsilon,
                      alpha = alpha,
                      gamma = gamma)
        
    def decay_epsilon(self):
        # Décroissance progressive d'epsilon
        min_epsilon = 0.01
        self.epsilon = max(min_epsilon, self.epsilon * 0.995)
        
    def save_q_table(self, filename=None):
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"q_table_{timestamp}.pkl"
        
        with open(filename, 'wb') as f:
            pickle.dump(self.q_table, f)
        print(f"Q-table sauvegardée dans {filename}")
    
    def load_q_table(self, filename):
        with open(filename, 'rb') as f:
            self.q_table = pickle.load(f)
        print(f"Q-table chargée depuis {filename}")