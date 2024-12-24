############################################################
# IMPORTS
# Required libraries for implementing Deep Q-Network (DQN)
############################################################
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import namedtuple, deque
import random

############################################################
# NEURAL NETWORK ARCHITECTURE
# Implements Dueling DQN architecture with:
# - Feature extraction layer
# - Separate value and advantage streams
# - Final Q-value computation
############################################################
class YahtzeeDQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        """
        Initialize DQN with dueling architecture
        Args:
            state_size: Dimension of input state (32)
            action_size: Number of possible actions (45)
            hidden_size: Size of hidden layers
        """
        super(YahtzeeDQN, self).__init__()
        
        # Shared feature extraction layers
        # Purpose: Initial processing of state input to extract meaningful features
        # - First linear layer maps input state to hidden representation
        # - Second linear layer further processes these features
        # - Dropout helps prevent overfitting by randomly deactivating neurons
        self.feature_layer = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1)  # Prevent overfitting
        )
        
        # Value stream - estimates state value
        # Purpose: Estimates how good the current state is, regardless of actions
        # - Processes shared features into a single value V(s)
        # - Gradually reduces dimensionality to output a scalar value
        # - This represents the intrinsic value of being in the current state
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Linear(hidden_size//2, 1)
        )
        
        # Advantage stream - estimates action advantages
        # Purpose: Estimates the relative advantage of each action in current state
        # - Processes shared features into advantage values for each possible action
        # - Gradually reduces dimensionality but maintains action_size outputs
        # - Represents how much better/worse each action is compared to others
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Linear(hidden_size//2, action_size)
        )
        
    def forward(self, state):
        """
        Compute Q-values using dueling architecture
        Returns Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        """
        features = self.feature_layer(state)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        q_values = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_values

############################################################
# AGENT IMPLEMENTATION
# Implements DQN agent with:
# - Experience replay
# - Target network
# - Epsilon-greedy exploration
# - Soft updates
############################################################
class YahtzeeAgent:
    def __init__(
        self,
        state_size,      # Dimension of the state space (dice + scorecard = 32)
        action_size,     # Total number of possible actions (32 reroll + 13 scoring = 45)
        game,           # Reference to the Yahtzee game instance
        hidden_size=256, # Number of neurons in hidden layers of the neural network
        batch_size=256, # Number of experiences to sample for each learning update
        learning_rate=1e-3,  # Step size for gradient descent optimization
        gamma=0.1,      # Discount factor for future rewards (lower because Yahtzee has short episodes)
        tau=1e-2,      # Soft update parameter for target network (1% update rate)
        memory_size=100000,  # Maximum number of experiences to store in replay buffer
        update_every=10     # Number of steps between network updates
    ):
        """
        Initialize DQN agent with all hyperparameters
        """
        self.state_size = state_size
        self.action_size = action_size
        self.game = game
        self.batch_size = batch_size
        self.gamma = gamma          
        self.tau = tau             
        self.update_every = update_every
        
        # Device configuration
        self.device = torch.device("mps")
        print(f"Using device: {self.device}")
        
        # Initialize Q-Networks (local and target)
        self.qnetwork_local = YahtzeeDQN(state_size, action_size, hidden_size).to(self.device)
        self.qnetwork_target = YahtzeeDQN(state_size, action_size, hidden_size).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.qnetwork_local.parameters(), 
            lr=learning_rate,
            weight_decay=1e-4  # L2 regularization
        )
        
        # Initialize replay memory
        self.memory = ReplayBuffer(action_size, memory_size, batch_size, self.device)
        self.t_step = 0
    
    ############################
    # EXPERIENCE COLLECTION
    ############################
    def step(self, state, action, reward, next_state, done):
        """
        Store experience in replay memory and trigger learning
        """
        # Add experience to memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every update_every time steps
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0 and len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences)
    
    ############################
    # ACTION SELECTION
    ############################
    def act(self, state, eps=0.):
        """
        Select action using epsilon-greedy policy
        """
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Evaluate action values
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
            
            # Apply action masking for invalid moves
            if self.game.rollLeft() == 0:
                action_values[:, :32] = float('-inf')  # Mask reroll actions
            
            # Mask completed categories
            completed_rows = self.game.getCompletedRows()
            for category in completed_rows:
                category_idx = list(self.game.scorecard.keys()).index(category)
                action_values[:, 32 + category_idx] = float('-inf')
        
        self.qnetwork_local.train()
        
        # Epsilon-greedy action selection
        if random.random() > eps:
            # Get valid actions
            valid_actions = []
            if self.game.rollLeft() > 0:
                valid_actions.extend(range(32))
            
            # Add available scoring categories
            for i, category in enumerate(self.game.scorecard.keys()):
                if category not in completed_rows:
                    valid_actions.append(32 + i)
            
            # Choose best valid action
            valid_action_values = action_values.cpu().data.numpy()[0]
            valid_action_values = [(i, val) for i, val in enumerate(valid_action_values) 
                                 if i in valid_actions]
            
            if valid_action_values:
                return max(valid_action_values, key=lambda x: x[1])[0]
            else:
                return random.choice(valid_actions)
        else:
            # Random valid action
            valid_actions = []
            if self.game.rollLeft() > 0:
                valid_actions.extend(range(32))
            
            for i, category in enumerate(self.game.scorecard.keys()):
                if category not in completed_rows:
                    valid_actions.append(32 + i)
            
            return random.choice(valid_actions)
    
    ############################
    # LEARNING PROCESS
    ############################
    def learn(self, experiences):
        """
        Update value parameters using batch of experience tuples
        """
        states, actions, rewards, next_states, dones = experiences
        
        # Get max predicted Q values for next states
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        
        # Compute Q targets for current states
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        
        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        
        # Compute loss and perform optimization
        loss = F.mse_loss(Q_expected, Q_targets)
        self.last_loss = loss.item()
        
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.qnetwork_local.parameters(), 1.0)
        
        self.optimizer.step()
        
        # Update target network
        self.soft_update(self.qnetwork_local, self.qnetwork_target)
    
    ############################
    # TARGET NETWORK UPDATE
    ############################
    def soft_update(self, local_model, target_model):
        """
        Soft update target model parameters:
        θ_target = τ*θ_local + (1 - τ)*θ_target
        """
        for target_param, local_param in zip(target_model.parameters(), 
                                           local_model.parameters()):
            target_param.data.copy_(
                self.tau * local_param.data + (1.0 - self.tau) * target_param.data
            )

############################################################
# EXPERIENCE REPLAY BUFFER
# Stores and samples experiences for training
############################################################
class ReplayBuffer:
    def __init__(self, action_size, buffer_size, batch_size, device):
        """Initialize replay buffer"""
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.device = device
        self.experience = namedtuple("Experience", 
                                   field_names=["state", "action", "reward", 
                                              "next_state", "done"])
    
    def add(self, state, action, reward, next_state, done):
        """Add new experience to memory"""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Random sample of experiences from memory"""
        experiences = random.sample(self.memory, k=self.batch_size)
        
        # Convert to torch tensors and move to correct device
        states = torch.FloatTensor(
            np.vstack([e.state for e in experiences if e is not None])
        ).to(self.device)
        actions = torch.LongTensor(
            np.vstack([e.action for e in experiences if e is not None])
        ).to(self.device)
        rewards = torch.FloatTensor(
            np.vstack([e.reward for e in experiences if e is not None])
        ).to(self.device)
        next_states = torch.FloatTensor(
            np.vstack([e.next_state for e in experiences if e is not None])
        ).to(self.device)
        dones = torch.FloatTensor(
            np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)
        ).to(self.device)
        
        return (states, actions, rewards, next_states, dones)
    
    def __len__(self):
        """Return current size of memory"""
        return len(self.memory) 