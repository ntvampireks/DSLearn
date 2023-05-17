from collections import deque
from model import DQN
import random
from environment import Field
import torch.optim as optim
import torch.nn as nn
import torch
import numpy as np
from rasterizer import Drawer
import time
from scipy.special import softmax


class Agent:

    def __init__(self, max_memory, batch_size, lr, epsilon=0.95, epsilon_rate=0.99, gamma=0.8):
        self.batch_size = batch_size
        self.lr = lr
        self.n_games = 0
        self.epsilon = epsilon
        self.epsilon_rate = epsilon_rate
        self.max_memory = max_memory
        self.gamma = gamma  # Коэффициент обесценивания
        self.memory = deque(maxlen=self.max_memory)
        self.model = DQN(16, 224, 4)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))  # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > self.batch_size:
            # случайно выбираем пачку из памяти
            mini_sample = random.sample(self.memory, self.batch_size)
        else:
            # или используем всю память
            mini_sample = self.memory
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        states = np.array(states)
        next_states = np.array(next_states)
        actions = np.array(actions)
        self.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.train_step(state, action, reward, next_state, done)

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        # для случая если на вход подали один-единственный кадр, а не пачку
        if len(state.shape) == 1:
            dn = list()
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            dn.append(done)
            done = dn
        # получили функцию ценности действий для текущего состояния/
        pred = self.model(state)
        target = pred.clone()
        for idx in range(len(done)):
            # считаем ценность действия для следующего состояния если текущий ход не ведет к проигрышу
            q_new = reward[idx]
            # вычислили прогноз ценности каждого из действий для следующего состояния
            # корректируем ценность действия на следующем шаге, можно переписать без циклов
            if not done[idx]:
                q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            target[idx][torch.argmax(action[idx]).item()] = q_new
        # теперь когда у нас есть предсказанное значение ценности за прошлое состояние и таргет можно корректировать
        # весовые коэффициенты
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()


# def softmax(x):
#    """Compute softmax values for each sets of scores in x."""
#    return np.exp(x) / np.sum(np.exp(x), axis=0)


def train(iterations, lr, memory_size, batch_size, epsilon, epsilon_rate, gamma):
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent(memory_size, batch_size, lr, epsilon, epsilon_rate, gamma)
    game = Field(20, 20, human=False)
    d = Drawer(600, 600, 4, "Snake", game)
    while agent.n_games < iterations:
        # Получаем прошлое состояние среды
        state_old = game.get_state()
        # вычисляем следующее действие

        if random.uniform(0, 1) > agent.epsilon:
            r = torch.from_numpy(np.array(state_old)).float()
            final_move = agent.model(r).detach().numpy()
        else:
            final_move = random.randint(0, 3)

        if type(final_move) == int:
            turn = final_move
            final_move = np.zeros(4)
            final_move[turn] = 1
        else:
            st = softmax(final_move)  # softmax(final_move) # вероятности каждого из возможных действий
            turn = np.random.choice(np.arange(0, 4), p=st)

        # выполняем ход, фиксируем новое состояние, награду, признак что игра завершена, общий счет на текущую игру
        state_new, reward, done, score = game.step(turn)
        state_new = np.array(state_new)
        # корректировка весов на основе последнего действия
        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        # закинули в нашу память результаты хода
        agent.remember(state_old, final_move, reward, state_new, done)
        # t = game.base

        d.refresh()
        d.update()
        # итерация закончена
        if done:
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            agent.epsilon = agent.epsilon * epsilon_rate
            if agent.n_games % 10 == 0:
                print('Game', agent.n_games, 'Score', score, 'Record:', record, "Totalscore:", total_score,
                      "Mean_score:", mean_score, "Epsilon:", agent.epsilon)


def play():
    agent = Agent(100000, 1000, 0.001)
    game = Field(20, 20, human=False)
    agent.model = torch.load("./model/model.pth")
    d = Drawer(600, 600, 4, "Snake", game)
    state = game.get_state()
    while True:
        r = torch.from_numpy(np.array(state)).float()
        final_move = agent.model(r).detach().numpy()
        turn = np.argmax(final_move).item()
        state, reward, done, score = game.step(turn)
        if done:
            game.reset()
            state = game.get_state()
        print("Reward:", reward, "Score:", score)
        d.refresh()
        d.update()
        time.sleep(0.05)
    d.mainloop()
