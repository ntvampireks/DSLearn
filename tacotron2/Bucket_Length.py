"""
Короткий алгоритм:

определяем границы корзинок
1. Выполняем проход по выборке, заполняем список пар<id, длина образца>
2. Индекс каждого полученного элемента сопоставляется соответствующей "корзинке",
   взято из batch_by_sequence_length библиотеки tensorflow
3. перемешиваем данные в каждой "корзинке"
4. формируем батчи из "корзинок"
5. перемешиваем батчи
6. выдергиваем поодному образцы из батчей (DataLoader использует индексы, а не списки)

Взято отсюда: https://gist.github.com/TrentBrick/bac21af244e7c772dc8651ab9c58328c
и немного доработано
Some code and inspiration taken from:
https://www.tensorflow.org/api_docs/python/tf/data/experimental/bucket_by_sequence_length

Данную реализацию доработал под обучение такотрона2
"""

import numpy as np
from random import shuffle
from torch.utils.data import Sampler
class BySequenceLengthSampler(Sampler):

    def __init__(self, data_source,
                bucket_boundaries, batch_size=64,):
        ind_n_len = []
        for i, p in enumerate(data_source):
            ind_n_len.append((i, p[1].shape[1]))
        self.ind_n_len = ind_n_len
        self.bucket_boundaries = bucket_boundaries
        self.batch_size = batch_size

    def __iter__(self):
        data_buckets = dict()
        # where p is the id number and seq_len is the length of this id number.
        for p, seq_len in self.ind_n_len:
            pid = self.element_to_bucket_id(p, seq_len)
            if pid in data_buckets.keys():
                data_buckets[pid].append(p)
            else:
                data_buckets[pid] = [p]

        for k in data_buckets.keys():

            data_buckets[k] = np.asarray(data_buckets[k])

        iter_list = []
        for k in data_buckets.keys():
            np.random.shuffle(data_buckets[k])
            iter_list += (np.array_split(data_buckets[k]
                           ,int(data_buckets[k].shape[0]/self.batch_size)
                             if int(data_buckets[k].shape[0]/self.batch_size) != 0 else 1))
        shuffle(iter_list)
        for i in iter_list:
            for n in i:
                yield n

    def __len__(self):
        return len(self.ind_n_len)

    def element_to_bucket_id(self, x, seq_length):
        boundaries = list(self.bucket_boundaries)
        buckets_min = [np.iinfo(np.int32).min] + boundaries
        buckets_max = boundaries + [np.iinfo(np.int32).max]
        conditions_c = np.logical_and(
          np.less_equal(buckets_min, seq_length),
          np.less(seq_length, buckets_max))
        bucket_id = np.min(np.where(conditions_c))
        return bucket_id

# использовать так:
#
#bucket_boundaries = [50,100,125,150,175,200,250,300]
#batch_sizes=32
#sampler = BySequenceLengthSampler(<your data>,bucket_boundaries, batch_sizes)

#dataloader = DataLoader(<your DataSet Object>, batch_size=1,
#                        batch_sampler=sampler,
#                        num_workers=0,
#                        drop_last=False, pin_memory=False)


""" 
As it is numpy functions you’ll need to keep it on the CPU for now. And as your BatchSampler already creates the batches, your DataLoader should have a batch size of 1.

Also, buckets for values smaller and larger than your buckets are also created so you won’t lose any data.

NB. Currently the batch size must be smaller than smallest number of sequences in any bucket so you may have to adjust your bucket boundaries depending on your batch sizes.
"""
