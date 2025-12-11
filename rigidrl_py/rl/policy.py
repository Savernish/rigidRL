"""
MLP Policy Network using rigidRL Tensor
Fully differentiable, works with AdamW optimizer
"""
import sys
import os
import math
import random

# Setup path to C++ module
script_dir = os.path.dirname(os.path.abspath(__file__))
core_dir = os.path.join(os.path.dirname(os.path.dirname(script_dir)), 'diff_sim_core')
os.add_dll_directory(core_dir)
sys.path.insert(0, core_dir)

import rigidRL as rigid


def init_weight(rows, cols):
    """Xavier initialization for weights"""
    std = math.sqrt(2.0 / (rows + cols))
    data = [random.gauss(0, std) for _ in range(rows * cols)]
    w = rigid.Tensor(rows, cols, True)
    for i in range(rows):
        for j in range(cols):
            w.set(i, j, data[i * cols + j])
    return w


def init_bias(size):
    """Zero initialization for biases"""
    return rigid.Tensor(size, 1, True)


class MLP:
    """Multi-Layer Perceptron using rigidRL Tensor"""
    
    def __init__(self, input_size, hidden_sizes, output_size, activation='tanh'):
        self.activation = activation
        self.layers = []
        
        sizes = [input_size] + hidden_sizes + [output_size]
        for i in range(len(sizes) - 1):
            w = init_weight(sizes[i+1], sizes[i])
            b = init_bias(sizes[i+1])
            self.layers.append((w, b))
    
    def forward(self, x):
        for i, (w, b) in enumerate(self.layers):
            x = w.matmul(x) + b
            if i < len(self.layers) - 1:
                if self.activation == 'tanh':
                    x = rigid.tanh(x)
                else:
                    x = rigid.relu(x)
        return x
    
    def get_action(self, state, action_scale=1.0):
        if isinstance(state, list):
            x = rigid.Tensor(state, False)
        else:
            x = state
        out = self.forward(x)
        action = rigid.tanh(out) * action_scale
        return action
    
    def parameters(self):
        params = []
        for w, b in self.layers:
            params.append(w)
            params.append(b)
        return params
