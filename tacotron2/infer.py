
import matplotlib.pylab as plt
import sys
import numpy as np
import torch
from train import Hparams
from train import load_model
from text import text_to_sequence
from denoiser import Denoiser

sys.path.append('waveglow/')
'''Простой пример инференса'''


def plot_data(data, figsize=(16, 4)):
    fig, axes = plt.subplots(1, len(data), figsize=figsize)
    for i in range(len(data)):
        axes[i].imshow(data[i], aspect='auto', origin='lower',
                       interpolation='none')


hparams = Hparams()
checkpoint_path = "checkpoint_10000"
model = load_model(hparams)
model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
_ = model.cuda().eval()

# по-умолчанию берет файлы конфига сети из текущей папки
waveglow_path = 'waveglow_2000'
waveglow = torch.load(waveglow_path)['model']
waveglow.cuda().eval()
for k in waveglow.convinv:
    k.float()
denoiser = Denoiser(waveglow)

text = "Hasta la vista, baby!"
sequence = np.array(text_to_sequence(text, ['english_cleaners']))[None, :]
sequence = torch.autograd.Variable(
    torch.from_numpy(sequence)).cuda().long()
voice = [0]
voice = np.array(voice)
voice = torch.autograd.Variable(torch.from_numpy(voice)).cuda().long()

mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence, voice)
plot_data((mel_outputs.float().data.cpu().numpy()[0],
           mel_outputs_postnet.float().data.cpu().numpy()[0],
           alignments.float().data.cpu().numpy()[0].T))
plt.savefig('foo1.png', bbox_inches='tight')
