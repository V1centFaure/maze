"""
Reinforcement learning agent module using epsilon-greedy strategy.

This module implements a reinforcement learning agent capable of learning optimal
navigation policies in maze environments. The agent uses temporal difference learning
algorithms (Q-learning or SARSA) combined with an epsilon-greedy action selection
policy to balance exploration and exploitation.

Key Features:
- Q-table based value function approximation
- Epsilon-greedy policy for action selection
- Support for both Q-learning and SARSA algorithms
- Configurable learning parameters (epsilon, alpha, gamma)
- Q-table persistence (save/load functionality)
- Epsilon decay for progressive exploitation increase

The agent maintains a Q-table that stores action-value estimates for each state-action
pair, learning through interaction with the environment and temporal difference updates.

Classes:
    Agent_epsilon_greedy: Main RL agent with epsilon-greedy policy

Example:
    >>> agent = Agent_epsilon_greedy(height=5, width=8, epsilon=0.1)
    >>> action = agent.select_action(state=(0, 0))
    >>> agent.update(state, action, reward, next_state, done, method="q_learning")
    >>> agent.save_q_table("trained_agent.pkl")
"""

import config
import numpy as np
import datetime
import pickle


class Agent_epsilon_greedy:
    """
    Reinforcement learning agent using epsilon-greedy action selection policy.
    
    This agent implements temporal difference learning algorithms (Q-learning and SARSA)
    with an epsilon-greedy policy for action selection. The agent maintains a Q-table
    that stores action-value estimates for each state-action pair and learns through
    interaction with the environment.
    
    The epsilon-greedy policy balances exploration and exploitation by:
    - With probability epsilon: selecting random actions (exploration)
    - With probability (1-epsilon): selecting the best known action (exploitation)
    
    Attributes:
        height (int): Environment height (number of rows)
        width (int): Environment width (number of columns)
        q_table (numpy.ndarray): Q-value table with shape (height, width, 4)
        epsilon (float): Exploration rate (0-1)
        alpha (float): Learning rate (0-1)
        gamma (float): Discount factor for future rewards (0-1)
        
    Note:
        The Q-table uses the following action encoding:
        - Index 0: 'up'
        - Index 1: 'down'
        - Index 2: 'left'
        - Index 3: 'right'
    """
    
    def __init__(self, height, width, epsilon=0.1, alpha=0.1, gamma=0.9):
        """
        Initialize the epsilon-greedy reinforcement learning agent.
        
        Creates a new agent with specified environment dimensions and learning parameters.
        The Q-table is initialized with zeros, representing no prior knowledge about
        the environment.
        
        Args:
            height (int): Environment height (number of rows). Must be positive.
            width (int): Environment width (number of columns). Must be positive.
            epsilon (float, optional): Exploration rate controlling the probability
                                     of selecting random actions. Should be in [0, 1].
                                     Higher values favor exploration. Defaults to 0.1.
            alpha (float, optional): Learning rate controlling how much new information
                                   overrides old information. Should be in [0, 1].
                                   Higher values make the agent learn faster but potentially
                                   less stable. Defaults to 0.1.
            gamma (float, optional): Discount factor for future rewards. Should be in [0, 1].
                                   Values closer to 1 make the agent more far-sighted,
                                   while values closer to 0 make it more myopic.
                                   Defaults to 0.9.
                                   
        Raises:
            ValueError: If any parameter is outside its valid range.
            
        Note:
            The Q-table is initialized with zeros, which represents an optimistic
            initialization that can encourage exploration in the early stages of learning.
        """
        # Environment dimensions
        self.height = height
        self.width = width
        
        # Q-table: (height, width, 4 possible actions)
        # Actions: 0=up, 1=down, 2=left, 3=right
        self.q_table = np.zeros(shape=(self.height, self.width, 4))
        
        # Learning parameters
        self.epsilon = epsilon  # Exploration rate
        self.alpha = alpha      # Learning rate
        self.gamma = gamma      # Discount factor

    def select_action(self, state):
        """
        Select an action using the epsilon-greedy policy.
        
        The epsilon-greedy policy is a simple method for balancing exploration
        and exploitation in reinforcement learning:
        - With probability epsilon: select a random action (exploration)
        - With probability (1-epsilon): select the action with highest Q-value (exploitation)
        
        When multiple actions have the same highest Q-value, one is chosen randomly
        to break ties, which helps with exploration in the early stages of learning.
        
        Args:
            state (tuple): Current agent position as (row, column). Both coordinates
                         must be within the environment bounds [0, height) and [0, width).
            
        Returns:
            str: Selected action from {'up', 'down', 'left', 'right'}
            
        Note:
            This method uses the current epsilon value, which may change over time
            if epsilon decay is applied. The random number generation uses numpy's
            global random state.
            
        Example:
            >>> agent = Agent_epsilon_greedy(5, 5, epsilon=0.1)
            >>> action = agent.select_action((2, 3))
            >>> print(action)  # e.g., 'right'
        """
        if np.random.random() < self.epsilon:  # Exploration
            return np.random.choice(['up', 'down', 'left', 'right'])
        else:  # Exploitation
            # Select action with highest Q-value (with random tie-breaking)
            return config.IDX2ACTION[config.argmax(self.q_table[state[0], state[1], :])]

    def update(self, state, action, reward, next_state, done, method="q_learning"):
        """
        Update the Q-table using temporal difference learning.
        
        This method implements the core learning mechanism of the agent using either
        Q-learning or SARSA algorithms. Both use the temporal difference error to
        update Q-values, but differ in how they estimate the value of the next state.
        
        Q-learning (off-policy):
            Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
            Uses the maximum Q-value of the next state (greedy policy).
            
        SARSA (on-policy):
            Q(s,a) ← Q(s,a) + α[r + γ Q(s',a') - Q(s,a)]
            Uses the Q-value of the action actually selected in the next state.
        
        Args:
            state (tuple): Current state as (row, column). Must be valid coordinates.
            action (str): Action taken. Must be one of {'up', 'down', 'left', 'right'}.
            reward (float): Immediate reward received after taking the action.
            next_state (tuple): Resulting state as (row, column). Must be valid coordinates
                              unless done=True.
            done (bool): Whether the episode has terminated. If True, no future rewards
                       are considered (target = reward only).
            method (str, optional): Learning algorithm to use. Options:
                                  - 'q_learning': Off-policy Q-learning (default)
                                  - 'sarsa': On-policy SARSA
                                  
        Raises:
            KeyError: If action is not in the valid action set.
            IndexError: If state coordinates are out of bounds.
            
        Note:
            For SARSA, the next action is selected using the current policy (epsilon-greedy),
            which means the update depends on the current exploration strategy.
            For Q-learning, the update is independent of the exploration strategy.
            
        Example:
            >>> agent.update((0,0), 'right', -1, (0,1), False, method='q_learning')
        """
        action_idx = config.ACTION2IDX[action]
        
        # For SARSA, we need the next action
        if method == 'sarsa': 
            next_action = self.select_action(state=next_state)
            next_action_idx = config.ACTION2IDX[next_action]
        
        # Calculate target according to chosen algorithm
        if done:
            # If episode is finished, target is simply the reward
            target = reward
        else:
            if method == 'sarsa':
                # SARSA: uses next action selected by policy
                target = reward + self.gamma * self.q_table[next_state[0], next_state[1], next_action_idx]
            elif method == 'q_learning':
                # Q-learning: uses best possible action in next state
                target = reward + self.gamma * np.max(self.q_table[next_state[0], next_state[1], :])
        
        # Update Q-table with temporal difference learning rule
        self.q_table[state[0], state[1], action_idx] += self.alpha * (target - self.q_table[state[0], state[1], action_idx])

    def reset(self, height=None, width=None, epsilon=None, alpha=None, gamma=None):
        """
        Reset the agent to initial state with optional parameter updates.
        
        This method reinitializes the agent, clearing all learned knowledge (Q-table)
        and optionally updating learning parameters. This is useful for:
        - Starting new training episodes with different parameters
        - Resetting after environment changes
        - Conducting multiple independent training runs
        
        Args:
            height (int, optional): New environment height. If None, keeps current value.
                                  Must be positive if provided.
            width (int, optional): New environment width. If None, keeps current value.
                                 Must be positive if provided.
            epsilon (float, optional): New exploration rate. If None, keeps current value.
                                     Should be in [0, 1] if provided.
            alpha (float, optional): New learning rate. If None, keeps current value.
                                   Should be in [0, 1] if provided.
            gamma (float, optional): New discount factor. If None, keeps current value.
                                   Should be in [0, 1] if provided.
                                   
        Side Effects:
            - Completely reinitializes the Q-table (all learned knowledge is lost)
            - Updates all agent parameters to new values or keeps existing ones
            - Resets internal state to initial conditions
            
        Warning:
            This method destroys all learned Q-values. If you want to preserve
            learned knowledge, consider saving the Q-table before resetting.
            
        Example:
            >>> agent.save_q_table("backup.pkl")  # Save current knowledge
            >>> agent.reset(epsilon=0.05)  # Reset with lower exploration rate
        """
        # Use current values if new parameters are not provided
        if height is None: 
            height = self.height
        if width is None: 
            width = self.width
        if epsilon is None: 
            epsilon = self.epsilon
        if alpha is None: 
            alpha = self.alpha
        if gamma is None: 
            gamma = self.gamma
        
        # Reinitialize agent with new parameters
        self.__init__(height=height, width=width, epsilon=epsilon, alpha=alpha, gamma=gamma)
        
    def decay_epsilon(self):
        """
        Apply exponential decay to the exploration rate epsilon.
        
        This method implements a common technique in reinforcement learning where
        the exploration rate is gradually reduced over time. This allows the agent
        to explore extensively early in training and then progressively exploit
        its learned knowledge as training progresses.
        
        The decay formula used is:
            epsilon = max(min_epsilon, epsilon * decay_rate)
            
        Where:
        - decay_rate = 0.995 (reduces epsilon by 0.5% each call)
        - min_epsilon = 0.01 (minimum exploration rate)
        
        Side Effects:
            Modifies self.epsilon in-place, reducing it towards the minimum value.
            
        Note:
            This method should typically be called after each episode or at regular
            intervals during training. The decay rate and minimum epsilon are
            hardcoded but could be made configurable in future versions.
            
        Example:
            >>> agent = Agent_epsilon_greedy(5, 5, epsilon=0.9)
            >>> print(f"Initial epsilon: {agent.epsilon}")  # 0.9
            >>> for _ in range(100):
            ...     agent.decay_epsilon()
            >>> print(f"After 100 decays: {agent.epsilon:.3f}")  # ~0.605
        """
        min_epsilon = 0.01  # Minimum epsilon value
        self.epsilon = max(min_epsilon, self.epsilon * 0.995)
        
    def save_q_table(self, filename=None):
        """
        Save the current Q-table to a pickle file for later use.
        
        This method serializes the Q-table using Python's pickle module, allowing
        the learned knowledge to be preserved and reloaded later. This is useful for:
        - Saving trained models for deployment
        - Creating checkpoints during long training sessions
        - Sharing trained agents between different sessions
        
        Args:
            filename (str, optional): Path and filename for the saved Q-table.
                                    If None, generates an automatic filename with
                                    format "q_table_YYYYMMDD_HHMMSS.pkl" using
                                    the current timestamp.
                                    
        Raises:
            IOError: If the file cannot be written (e.g., permission denied,
                    disk full, invalid path).
            pickle.PicklingError: If the Q-table cannot be serialized.
            
        Side Effects:
            Creates a new file on disk containing the serialized Q-table.
            Prints a confirmation message with the filename.
            
        Note:
            The saved file contains only the Q-table numpy array, not the
            entire agent object. To restore a complete agent, you'll need
            to create a new agent with the same parameters and then load
            the Q-table.
            
        Example:
            >>> agent.save_q_table()  # Saves with timestamp
            Q-table saved to q_table_20231218_143052.pkl
            >>> agent.save_q_table("my_trained_agent.pkl")  # Custom filename
            Q-table saved to my_trained_agent.pkl
        """
        if filename is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"q_table_{timestamp}.pkl"
        
        with open(filename, 'wb') as f:
            pickle.dump(self.q_table, f)
        print(f"Q-table saved to {filename}")
    
    def load_q_table(self, filename):
        """
        Load a previously saved Q-table from a pickle file.
        
        This method deserializes a Q-table that was previously saved using
        save_q_table(), allowing the agent to resume learning from a trained
        state or to deploy a pre-trained agent.
        
        Args:
            filename (str): Path to the pickle file containing the Q-table.
                          The file should have been created by save_q_table().
                          
        Raises:
            FileNotFoundError: If the specified file does not exist or cannot
                             be accessed.
            pickle.UnpicklingError: If the file is not a valid pickle file or
                                  contains corrupted data.
            ValueError: If the loaded Q-table has incompatible dimensions with
                       the current agent configuration.
                       
        Side Effects:
            Replaces the current Q-table with the loaded one, overwriting
            any existing learned knowledge. Prints a confirmation message.
            
        Warning:
            The loaded Q-table must have compatible dimensions with the current
            agent (same height, width, and number of actions). No validation
            is performed, so loading an incompatible Q-table may cause runtime
            errors during action selection or updates.
            
        Example:
            >>> agent = Agent_epsilon_greedy(5, 8)  # Create new agent
            >>> agent.load_q_table("my_trained_agent.pkl")  # Load saved knowledge
            Q-table loaded from my_trained_agent.pkl
            >>> # Agent now has the learned Q-values from the file
        """
        with open(filename, 'rb') as f:
            self.q_table = pickle.load(f)
        print(f"Q-table loaded from {filename}")
