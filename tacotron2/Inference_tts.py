import sys
sys.path.append('waveglow/')
import numpy as np
import torch
from train import Hparams
from model import Tacotron2
from layers import TacotronSTFT, STFT
from audio_processing import griffin_lim
from train import load_model
from text import text_to_sequence, text_to_phonemes_to_sequence, _clean_text
from denoiser import Denoiser
import time
import os
from scipy.io.wavfile import write
from pydub import AudioSegment, effects
from phonemizer import phonemize




def load_models(tacotron2_file: str, waveglow_file: str):
    hparams = Hparams()
    checkpoint_path = tacotron2_file
    model = load_model(hparams)
    model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
    _ = model.cuda().eval();

    waveglow_path = waveglow_file #'waveglow_256channels.pt'
    waveglow = torch.load(waveglow_path)['model']
    waveglow.cuda().eval()
    for k in waveglow.convinv:
        k.float()
    return model, waveglow


def text_to_speak(tacotron, waveglow, phrase, dict_id, filename=None):
    if not filename:
        filename = str(time.time())
    hparams = Hparams()

    denoiser = Denoiser(waveglow)

    text = _clean_text(phrase, ['english_cleaners'])

    text = phonemize(text, language='en-us', backend='espeak', preserve_punctuation=True, njobs=8)
    sequence = np.array(text_to_phonemes_to_sequence(text))[None, :]

    sequence = torch.autograd.Variable(
        torch.from_numpy(sequence)).cuda().long()

    voice = [int(dict_id)]
    voice = np.array(voice)
    voice = torch.autograd.Variable(torch.from_numpy(voice)).cuda().long()

    mel_outputs, mel_outputs_postnet, _, alignments = tacotron.inference(sequence, voice)
    with torch.no_grad():
        audio = waveglow.infer(mel_outputs_postnet, sigma=0.666)

    audio = denoiser(audio, strength=0.01)[:, 0]


    #audio = audio.cpu().numpy()
    audio = audio[0].data.cpu().numpy().astype(np.float32)
    audio_path = f"{filename}.wav"
    save_path = os.path.join('wavs', audio_path)
    write(save_path, hparams.sampling_rate, audio)
    # normalize volume
    print("audio saved at: {}".format(save_path))

    return audio_path