import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.autograd import Variable
from torch import nn
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def init_hidden(x: torch.Tensor, hidden_size: int, num_dir: int = 1, xavier: bool = True):
    """
    Initialize hidden.
    Args:
        x: (torch.Tensor): input tensor
        hidden_size: (int):
        num_dir: (int): number of directions in LSTM
        xavier: (bool): wether or not use xavier initialization
    """
    if xavier:
        return nn.init.xavier_normal_(torch.zeros(num_dir, x.size(0), hidden_size)).to(device)
    return Variable(torch.zeros(num_dir, x.size(0), hidden_size)).to(device)

class Encoder(nn.Module):
    def __init__(self, input_len, input_size, hidden_size):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.input_len = input_len
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=self.hidden_size)

    def forward(self, x):
        h_t, c_t = (init_hidden(x, self.hidden_size),
                    init_hidden(x, self.hidden_size))
        # кодируем в пространство B x Hidden_Size x Input_length
        input_encoded = Variable(torch.zeros(x.size(0), self.hidden_size, self.input_len))
        for t in range(self.input_len):
            # unsqueeze в данном случае добавляет размерность к x[:, :, t],
            _, (h_t, c_t) = self.lstm(x[:, :, t].unsqueeze(0), (h_t, c_t))
            # добавляем результат последнего хиддена LSTM
            input_encoded[:, :, t] = h_t
        # на выходе тензор размерности Batch X Hidden_Size X Input_length(время)
        return _, input_encoded


# (2) Decoder"""


class Decoder(nn.Module):
    def __init__(self, encoder_hidden_size, decoder_hidden_size, input_len, output_len ):
        """
        Initialize the network.
        Args:
            config:
        """
        super(Decoder, self).__init__()
        self.input_len = input_len
        self.output_len = output_len
        self.decoder_hidden_size = decoder_hidden_size
        self.encoder_hidden_size = encoder_hidden_size

        self.lstm = nn.LSTM(self.encoder_hidden_size, self.decoder_hidden_size, bidirectional=False)
        self.fc = nn.Linear(self.decoder_hidden_size*self.input_len, self.output_len)

    def forward(self, x):
        h_t, c_t = (init_hidden(x, self.decoder_hidden_size),
                    init_hidden(x, self.decoder_hidden_size))
        # BxHxT
        input_encoded = Variable(torch.zeros(x.size(0), self.decoder_hidden_size, self.input_len)).to(device)

        for t in range(self.input_len):
            lstm_out, (h_t, c_t) = self.lstm(x[:,:,t].unsqueeze(0).to(device), (h_t, c_t))
            input_encoded[:,:,t] = lstm_out.squeeze(0)

        # Bx(H*T)
        return self.fc(input_encoded.view(x.size(0), self.decoder_hidden_size*self.input_len))


# (3) Autoencoder : putting the encoder and decoder together
class LSTM_AE(nn.Module):
    def __init__(self, input_len, input_size, embedding_dim, output_len, output_size):
        super().__init__()

        self.input_len = input_len
        self.input_size = input_size
        self.embedding_dim = embedding_dim
        self.output_len = output_len
        self.output_size = output_size
        self.encoder = Encoder(self.input_len, self.input_size, self.embedding_dim)
        self.decoder = Decoder(self.embedding_dim, self.embedding_dim, self.input_len, self.output_len)

    def forward(self, x):
        torch.manual_seed(0)

        _, encoded = self.encoder(x)

        decoded = self.decoder(encoded)
        return encoded, decoded