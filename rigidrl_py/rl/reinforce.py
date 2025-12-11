"""
REINFORCE Algorithm with Exploration
Using rigidRL Tensor and AdamW optimizer
"""
import sys
import os
import math
import random

script_dir = os.path.dirname(os.path.abspath(__file__))
core_dir = os.path.join(os.path.dirname(os.path.dirname(script_dir)), 'diff_sim_core')
os.add_dll_directory(core_dir)
sys.path.insert(0, core_dir)

import rigidRL as rigid
from .policy import MLP


class REINFORCE:
    """REINFORCE with exploration noise"""
    
    def __init__(self, state_dim, action_dim, hidden_sizes=[64, 64], 
                 lr=0.001, gamma=0.99, action_scale=1.0,
                 exploration_start=1.0, exploration_end=0.1, exploration_decay=0.995):
        self.gamma = gamma
        self.action_scale = action_scale
        
        # Exploration parameters
        self.exploration = exploration_start
        self.exploration_end = exploration_end
        self.exploration_decay = exploration_decay
        
        self.policy = MLP(state_dim, hidden_sizes, action_dim, activation='tanh')
        self.optimizer = rigid.AdamW(self.policy.parameters(), lr=lr)
        
        self.states = []
        self.actions_data = []
        self.rewards = []
    
    def select_action(self, state, training=True):
        """Select action with exploration noise during training"""
        action = self.policy.get_action(state, self.action_scale)
        
        # Add exploration noise during training
        if training and self.exploration > 0.01:
            noise_scale = self.exploration * self.action_scale
            noisy_values = []
            for i in range(action.rows()):
                base_val = action.data[i, 0]
                noise = random.gauss(0, noise_scale)
                noisy_values.append(base_val + noise)
            
            # Store state and noisy action
            self.states.append(state if isinstance(state, list) else list(state))
            self.actions_data.append(noisy_values)
            
            # Return noisy action as tensor
            return rigid.Tensor(noisy_values, False)
        else:
            self.states.append(state if isinstance(state, list) else list(state))
            self.actions_data.append([action.data[i, 0] for i in range(action.rows())])
            return action
    
    def store_reward(self, reward):
        self.rewards.append(reward)
    
    def compute_returns(self):
        returns = []
        G = 0
        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        
        if len(returns) > 1:
            mean_r = sum(returns) / len(returns)
            std_r = math.sqrt(sum((r - mean_r)**2 for r in returns) / max(len(returns), 1))
            if std_r > 1e-8:
                returns = [(r - mean_r) / std_r for r in returns]
        
        return returns
    
    def update(self):
        """Update policy and decay exploration"""
        if len(self.rewards) == 0:
            return 0.0
        
        self.optimizer.zero_grad()
        returns = self.compute_returns()
        
        n_samples = min(15, len(self.states))
        step_size = max(1, len(self.states) // n_samples)
        
        total_loss = rigid.Tensor([0.0])
        
        for i in range(0, len(self.states), step_size):
            if i >= len(returns):
                break
            state = self.states[i]
            G = returns[i]
            
            action = self.policy.get_action(state, self.action_scale)
            action_loss = action * (-G)
            total_loss = total_loss + action_loss.sum()
        
        total_loss.backward()
        self.optimizer.step()
        
        # Decay exploration
        self.exploration = max(self.exploration_end, self.exploration * self.exploration_decay)
        
        loss_val = total_loss.data[0, 0]
        self.states = []
        self.actions_data = []
        self.rewards = []
        
        return loss_val
    
    def get_exploration(self):
        return self.exploration
