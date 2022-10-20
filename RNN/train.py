import pandas as pd
from torch import nn
import torch
from torch.utils.data import DataLoader
from models import LSTM_AE
from dataset import RNDataset


cuda_device = 0
device = f'cuda:{cuda_device}' if cuda_device != -1 else 'cpu'

drop_cols = ['MEASURED_IN_DATE','IS_GTM', 'P_ZAB', 'WELL_NAME', 'SPLIT','WATER_CUT']
df = pd.read_csv('F:\\test_data.csv')
train = RNDataset(df, 'LIQ_RATE', drop_cols, )
valid = RNDataset(df, 'LIQ_RATE', drop_cols, is_train=False)

X, y = train[1]
trainLoader = DataLoader(train, batch_size=100, shuffle=False, drop_last=True)
valLoader = DataLoader(valid, batch_size=30, shuffle=False, drop_last=True)
cuda_device = 0

torch.manual_seed(0)
model = LSTM_AE(200, 14, 128, 120, 1).to(device) #LSTMModel(200, 128, 2, 120).to(device)
optim = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

for epoch in range(100):
    losses = []
    #optim.zero_grad()
    for i, (X, y) in enumerate(trainLoader):
        optim.zero_grad()
        _, predict = model(X.to(device).float())

        loss = criterion(predict, y.to(device).float())
        losses.append(loss.item())
        loss.backward()

        optim.step()

    print(sum(losses))

for i, (X, y) in enumerate(valLoader):
    with torch.no_grad():
        _, predict = model(X.to(device).float())
        loss = criterion(predict, y.to(device).float())
        print("Full_Error:" , loss.item())


print()