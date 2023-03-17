from rasterizer import Drawer
from environment import Field
import time
import torch
from agent import Agent, train, play
import numpy as np
X_SIZE = 20
Y_SIZE = 20


if __name__ == '__main__':
    train(10000, 0.001, 5000, 123)
    # play()


