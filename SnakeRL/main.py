from rasterizer import Drawer
from environment import Field
import time
import torch
from agent import Agent, train, play
import numpy as np
X_SIZE = 20
Y_SIZE = 20


if __name__ == '__main__':
    train(iterations=10000, lr=0.001, memory_size=50000, batch_size=123, epsilon=0.95, epsilon_rate=0.99, gamma=0.8)
    # play()


