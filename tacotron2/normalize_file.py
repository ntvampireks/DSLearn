import random
import numpy as np
import torch
import torch.utils.data
import csv
import layers
from utils import load_wav_to_torch, load_filepaths_and_text
from text import text_to_sequence, \
                _symbols_to_sequence, _phonemes_to_sequence, text_to_phonemes_to_sequence, _clean_text
from phonemizer import phonemize


text_cleaners = ['english_cleaners']
audiopaths_and_text = load_filepaths_and_text('D:\\test.json')
i=0
#for lst in audiopaths_and_text:
#    t = _clean_text(lst[1], text_cleaners)
#    t = phonemize(t, language='en-us', backend='espeak', preserve_punctuation=True)
#    audiopaths_and_text[i].append(t)
#    i = i + 1
b = [el[1] for el in audiopaths_and_text]
L = phonemize(b)

for lst in audiopaths_and_text:
    audiopaths_and_text[i].append(L[i])
    i = i + 1

fields = ['Path', 'Phrase', 'Dict_id', 'Transcript']
with open('D:\\GFG', 'w',  encoding='utf8') as f:
    # using csv.writer method from CSV package
    write = csv.writer(f, delimiter='|',)

    write.writerow(fields)
    write.writerows(audiopaths_and_text)

print(0)