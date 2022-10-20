import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, step, single_step=False):
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i - history_size, i, step)
        data.append(dataset[indices])

        if single_step:
            labels.append(target[i + target_size])

        else:
            labels.append(target[i:i + target_size])
    return np.transpose(np.array(data), (0, 2, 1)), np.array(labels)


class RNDataset(Dataset):

    def __init__(self, df, target_col, cols_to_drop=[], is_train=True,
                 past_history=200, future_target=120, step=1, train_split=300):
        self.df = df.copy()

        self.df['MEASURED_IN_DATE'] = pd.to_datetime(self.df['MEASURED_IN_DATE'])
        self.df['Date_dd_INTAKE'] = self.df.MEASURED_IN_DATE.dt.day
        self.df['Date_mm_INTAKE'] = self.df.MEASURED_IN_DATE.dt.month
        self.df['Date_y_INTAKE'] = self.df.MEASURED_IN_DATE.dt.year
        self.df['Date_weekday_INTAKE'] = self.df.MEASURED_IN_DATE.dt.weekday
        self.df['Date_dy_INTAKE'] = self.df.MEASURED_IN_DATE.dt.dayofyear
        self.df['Date_HH_INTAKE'] = self.df.MEASURED_IN_DATE.dt.hour

        self.df.drop(cols_to_drop, axis=1, inplace=True)

        t = self.df[target_col].values
        # print(t)
        self.df.drop([target_col], axis=1, inplace=True)
        dataset = self.df.values
        scaler = StandardScaler().fit(dataset)

        dataset = scaler.transform(dataset)

        if is_train:
            self.x, self.y = multivariate_data(dataset, t, 0,
                                 train_split, past_history,
                                 future_target, step,
                                 single_step=False)

        else:
            self.x, self.y = multivariate_data(dataset, t,
                                               train_split, None, past_history,
                                               future_target, step,
                                               single_step=False)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]