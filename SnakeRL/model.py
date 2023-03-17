import torch
import torch.nn as nn
import os


class DQN(nn.Module):

    def __init__(self, in_channel, hidden_dim, action_size):
        super().__init__()
        # self.conv1 = nn.Conv2d(in_channel, hidden_dim, kernel_size=kernel_size, stride=1, padding=padding)
        # s#elf.conv2 = nn.Conv2d(hidden_dim, 50, kernel_size=21, padding=10, stride=1)

        # self.conv3 = nn.Conv2d(50, hidden_dim, kernel_size=21, padding=10, stride=1)

        self.linear1 = nn.Linear(in_channel, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, action_size)

        self.act = nn.ReLU()

    def forward(self, x):
        # x = self.act(self.conv1(x))
        # x = self.act(self.conv2(x))
        # x = self.act(self.conv3(x))
        # x# = torch.flatten(x,1)
        x = self.act(self.linear1(x))
        return self.linear2(x)

    def save(self, file_name="model.pth"):
        model_folder_path = "./model"
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        filename = os.path.join(model_folder_path, file_name)
        torch.save(self, filename)


class Conv_DQN(nn.Module):

    def __init__(self, in_channel, hidden_dim, action_size):
        super().__init__()
        # self.conv1 = nn.Conv2d(in_channel, hidden_dim, kernel_size=kernel_size, stride=1, padding=padding)
        # s#elf.conv2 = nn.Conv2d(hidden_dim, 50, kernel_size=21, padding=10, stride=1)

        # self.conv3 = nn.Conv2d(50, hidden_dim, kernel_size=21, padding=10, stride=1)

        self.conv1 = nn.Conv2d(1, hidden_dim, kernel_size=21, stride=1, padding=10)


        self.linear2 = nn.Linear(hidden_dim, action_size)

        self.act = nn.ReLU()

    def forward(self, field, state):
        # x = self.act(self.conv1(x))
        # x = self.act(self.conv2(x))
        # x = self.act(self.conv3(x))
        # x# = torch.flatten(x,1)
        x = self.conv1(field)
        x = x.concat(state)

        x = self.act(self.linear1(x))
        return self.linear2(x)

    def save(self, file_name="model.pth"):
        model_folder_path = "./model"
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        filename = os.path.join(model_folder_path, file_name)
        torch.save(self, filename)


