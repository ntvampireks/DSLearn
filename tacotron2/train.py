import os
import time
import argparse
import math
from numpy import finfo
from text import symbols
import torch
from distributed import apply_gradient_allreduce
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
#import espeak_phonemizer
from model import Tacotron2
from data_utils import TextMelLoader, TextMelCollate
from loss_function import Tacotron2Loss
from logger import Tacotron2Logger
from Bucket_Length import BySequenceLengthSampler
#  from hparams import create_hparams
#import netron
import torch.onnx

class Hparams:
    def __init__(self):
        ################################
        # Experiment Parameters        #
        ################################
        self.epochs = 500
        self.iters_per_checkpoint = 500
        self.seed = 1234
        self.dynamic_loss_scaling: bool = True
        self.fp16_run:bool = False
        self.distributed_run: bool = False
        self.dist_backend: str = "nccl"
        self.dist_url = "tcp://localhost:54321"
        self.cudnn_enabled: bool = True
        self.cudnn_benchmark: bool = False
        self.ignore_layers = ['embedding.weight']

        ################################
        # Data Parameters             #
        ################################
        self.load_mel_from_disk: bool = False
        self.training_files = 'D:\\train.json'
        self.validation_files = 'D:\\test.json'
        self.text_cleaners = ['english_cleaners']
        self.use_phonemes = True
        ################################
        # Audio Parameters             #
        ################################
        self.max_wav_value = 32768.0
        self.sampling_rate = 16000
        self.filter_length = 1024
        self.hop_length = 256
        self.win_length = 1024
        self.n_mel_channels = 80
        self.mel_fmin = 0.0
        self.mel_fmax = 8000.0

        ################################
        # Model Parameters             #
        ################################
        self.n_symbols = len(symbols)
        self.symbols_embedding_dim = 512

        # Encoder parameters
        self.encoder_kernel_size = 5
        self.encoder_n_convolutions = 3
        self.encoder_embedding_dim = 512

        #размер эмбеддинга голосов
        self.multivoice_embedding_dim = 32
        self.multivoice_max_voices = 10
        self.enc_aft_concat_voice_dim = self.encoder_embedding_dim + self.multivoice_embedding_dim

        # Decoder parameters
        self.n_frames_per_step = 1  # currently only 1 is supported
        self.decoder_rnn_dim = 1024 #+ int(self.multivoice_embedding_dim * 2)
        self.prenet_dim = 256 #+ int(self.multivoice_embedding_dim/2)
        self.max_decoder_steps = 1500
        self.gate_threshold = 0.5
        self.p_attention_dropout = 0.1
        self.p_decoder_dropout = 0.1

        # Attention parameters
        self.attention_rnn_dim = 1024 #+ int(self.multivoice_embedding_dim*2)
        self.attention_dim = 128

        # Location Layer parameters
        self.attention_location_n_filters = 32
        self.attention_location_kernel_size = 31

        # Mel-post processing network parameters
        self.postnet_embedding_dim = 512
        self.postnet_kernel_size = 5
        self.postnet_n_convolutions = 5

        ################################
        # Optimization Hyperparameters #
        ################################
        self.use_saved_learning_rate:bool = False
        self.learning_rate = 0.0005
        self.weight_decay = 1e-6
        self.grad_clip_thresh = 1.0
        self.batch_size = 48
        self.mask_padding = True  # set model's padded outputs to padded values



def reduce_tensor(tensor, n_gpus):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= n_gpus
    return rt


def init_distributed(hparams, n_gpus, rank, group_name):
    assert torch.cuda.is_available(), "Distributed mode requires CUDA."
    print("Initializing Distributed")

    # Set cuda device so everything is done on the right GPU.
    torch.cuda.set_device(rank % torch.cuda.device_count())

    # Initialize distributed communication
    dist.init_process_group(
        backend=hparams.dist_backend, init_method=hparams.dist_url,
        world_size=n_gpus, rank=rank, group_name=group_name)
    print("Done initializing distributed")


def prepare_dataloaders(hparams):
    # Get data, data loaders and collate function ready
    trainset = TextMelLoader(hparams.training_files, hparams,)
    valset = TextMelLoader(hparams.validation_files, hparams)
    collate_fn = TextMelCollate(hparams.n_frames_per_step)

    if hparams.distributed_run:
        train_sampler = DistributedSampler(trainset)
        shuffle = False
    else:
        train_sampler = None
        shuffle = False
        bucket_boundaries = [50, 100, 150, 200, 300, 400, 600, 800] #[100, 200, 300, 400, 500, 600, 700, 800, 900, 1000 ]
        batch_sizes = hparams.batch_size
        train_sampler = BySequenceLengthSampler(trainset, bucket_boundaries, batch_sizes)

        #batch_size = 1,
        #batch_sampler = sampler,
        #num_workers = 0,
        #drop_last = False, pin_memory = False)

    train_loader = DataLoader(trainset, num_workers=0, shuffle=shuffle,
                              sampler=train_sampler,
                              batch_size=48, pin_memory=False,
                              drop_last=True, collate_fn=collate_fn)
    return train_loader, valset, collate_fn


def prepare_directories_and_logger(output_directory, log_directory, rank):
    if rank == 0:
        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)
            os.chmod(output_directory, 0o775)
        logger = Tacotron2Logger(os.path.join(output_directory, log_directory))
    else:
        logger = None
    return logger


def load_model(hparams):
    model = Tacotron2(hparams).cuda()
    if hparams.fp16_run:
        model.decoder.attention_layer.score_mask_value = finfo('float16').min

    if hparams.distributed_run:
        model = apply_gradient_allreduce(model)

    return model


def warm_start_model(checkpoint_path, model, ignore_layers):
    assert os.path.isfile(checkpoint_path)
    print("Warm starting model from checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model_dict = checkpoint_dict['state_dict']
    if len(ignore_layers) > 0:
        model_dict = {k: v for k, v in model_dict.items()
                      if k not in ignore_layers}
        dummy_dict = model.state_dict()
        dummy_dict.update(model_dict)
        model_dict = dummy_dict
    model.load_state_dict(model_dict)
    return model


def load_checkpoint(checkpoint_path, model, optimizer):
    assert os.path.isfile(checkpoint_path)
    print("Loading checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint_dict['state_dict'])
    optimizer.load_state_dict(checkpoint_dict['optimizer'])
    learning_rate = checkpoint_dict['learning_rate']
    iteration = checkpoint_dict['iteration']
    print("Loaded checkpoint '{}' from iteration {}" .format(
        checkpoint_path, iteration))
    return model, optimizer, learning_rate, iteration


def save_checkpoint(model, optimizer, learning_rate, iteration, filepath):
    print("Saving model and optimizer state at iteration {} to {}".format(
        iteration, filepath))
    torch.save({'iteration': iteration,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'learning_rate': learning_rate}, filepath)


def validate(model, criterion, valset, iteration, batch_size, n_gpus,
             collate_fn, logger, distributed_run, rank):
    """Handles all the validation scoring and printing"""
    bucket_boundaries = [50, 100, 150, 200, 300, 400, 600, 800]  # [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000 ]
    batch_sizes = hparams.batch_size
   # sampler = BySequenceLengthSampler(valset, bucket_boundaries, batch_sizes)
    model.eval()
    with torch.no_grad():
        val_sampler = DistributedSampler(valset) if distributed_run else BySequenceLengthSampler(valset,
                                                                                                 bucket_boundaries,
                                                                                                 batch_sizes)
        val_loader = DataLoader(valset, sampler=val_sampler, num_workers=0,
                                shuffle=False, batch_size=batch_size,
                                pin_memory=False, collate_fn=collate_fn)

        val_loss = 0.0
        for i, batch in enumerate(val_loader):
            x, y = model.parse_batch(batch)
            y_pred = model(x)
            loss = criterion(y_pred, y)
            if distributed_run:
                reduced_val_loss = reduce_tensor(loss.data, n_gpus).item()
            else:
                reduced_val_loss = loss.item()
            val_loss += reduced_val_loss
        val_loss = val_loss / (i + 1)

    model.train()
    if rank == 0:
        print("Validation loss {}: {:9f}  ".format(iteration, val_loss))
        logger.log_validation(val_loss, model, y, y_pred, iteration)


def train(output_directory, log_directory, checkpoint_path, warm_start, n_gpus,
          rank, group_name, hparams):
    """Training and validation logging results to tensorboard and stdout

    Params
    ------
    output_directory (string): directory to save checkpoints
    log_directory (string) directory to save tensorboard logs
    checkpoint_path(string): checkpoint path
    n_gpus (int): number of gpus
    rank (int): rank of current gpu
    hparams (object): comma separated list of "name=value" pairs.
    """
    hparams = Hparams()

    if hparams.distributed_run:                                             #инициализируем распределенное приложение
        init_distributed(hparams, n_gpus, rank, group_name)
        init_distributed(hparams, n_gpus, rank, group_name)

    torch.manual_seed(hparams.seed)                                         #задаем начельные seed
    torch.cuda.manual_seed(hparams.seed)

    model = load_model(hparams)                                             #инициализируем модель
    learning_rate = hparams.learning_rate                                   #задаем скорость обучения
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,      #задаем оптимизатор
                                 weight_decay=hparams.weight_decay)

    if hparams.fp16_run:                                                    #не вполне ясно что такое
        from apex import amp
        model, optimizer = amp.initialize(
            model, optimizer, opt_level='O2')

    if hparams.distributed_run:                                             #для распред вычислений, разберу позднее
        model = apply_gradient_allreduce(model)

    criterion = Tacotron2Loss()                                             #задали функцию потерь

    logger = prepare_directories_and_logger(
        output_directory, log_directory, rank)                              #настройки логирования

    train_loader, valset, collate_fn = prepare_dataloaders(hparams)

    # Load checkpoint if one exists
    iteration = 0
    epoch_offset = 0
    if checkpoint_path is not None:
        if warm_start:
            model = warm_start_model(
                checkpoint_path, model, hparams.ignore_layers)
        else:
            model, optimizer, _learning_rate, iteration = load_checkpoint(
                checkpoint_path, model, optimizer)
            if hparams.use_saved_learning_rate:
                learning_rate = _learning_rate
            iteration += 1  # next iteration is iteration + 1
            epoch_offset = max(0, int(iteration / len(train_loader)))

    model.train()
    is_overflow = False
    # ================ MAIN TRAINNIG LOOP! ===================
    for epoch in range(epoch_offset, hparams.epochs):
        print("Epoch: {}".format(epoch))
        if not is_overflow:
            validate(model, criterion, valset, iteration,
                     hparams.batch_size, n_gpus, collate_fn, logger,
                     hparams.distributed_run, rank)
            if rank == 0:
                checkpoint_path = os.path.join(
                    output_directory, "checkpoint_epoch_{}".format(epoch))
                save_checkpoint(model, optimizer, learning_rate, iteration,
                                checkpoint_path)

        for i, batch in enumerate(train_loader):
            start = time.perf_counter()
            #for param_group in optimizer.param_groups:
            #    param_group['lr'] = learning_rate
            model.zero_grad()
            x, y = model.parse_batch(batch)
            y_pred = model(x)

            loss = criterion(y_pred, y)
            if hparams.distributed_run:
                reduced_loss = reduce_tensor(loss.data, n_gpus).item()
            else:
                reduced_loss = loss.item()
            if hparams.fp16_run:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if hparams.fp16_run:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    amp.master_params(optimizer), hparams.grad_clip_thresh)
                is_overflow = math.isnan(grad_norm)
            else:
                try:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), hparams.grad_clip_thresh, error_if_nonfinite=True)

                except:
                    print("error on batch {}".format(i))
            optimizer.step()

            if not is_overflow and rank == 0:
                duration = time.perf_counter() - start
                print("Train loss {} {:.6f} Grad Norm {:.6f} {:.2f}s/it".format(
                    iteration, reduced_loss, grad_norm, duration))
                logger.log_training(
                    reduced_loss, grad_norm, learning_rate, duration, iteration)

            if not is_overflow and (iteration % hparams.iters_per_checkpoint == 0):
                validate(model, criterion, valset, iteration,
                         hparams.batch_size, n_gpus, collate_fn, logger,
                         hparams.distributed_run, rank)
                if rank == 0:
                    checkpoint_path = os.path.join(
                        output_directory, "checkpoint_{}".format(iteration))
                    save_checkpoint(model, optimizer, learning_rate, iteration,
                                    checkpoint_path)

            iteration += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_directory', type=str,
                        help='directory to save checkpoints')
    parser.add_argument('-l', '--log_directory', type=str,
                        help='directory to save tensorboard logs')
    parser.add_argument('-c', '--checkpoint_path', type=str, default=None,
                        required=False, help='checkpoint path')
    parser.add_argument('--warm_start', action='store_true',
                        help='load model weights only, ignore specified layers')
    parser.add_argument('--n_gpus', type=int, default=1,
                        required=False, help='number of gpus')
    parser.add_argument('--rank', type=int, default=0,
                        required=False, help='rank of current gpu')
    parser.add_argument('--group_name', type=str, default='group_name',
                        required=False, help='Distributed group name')
    parser.add_argument('--hparams', type=str,
                        required=False, help='comma separated name=value pairs')

    args = parser.parse_args()
    hparams = Hparams()

    torch.backends.cudnn.enabled = bool(hparams.cudnn_enabled)
    torch.backends.cudnn.benchmark = bool(hparams.cudnn_benchmark)

    print("FP16 Run:", hparams.fp16_run)
    print("Dynamic Loss Scaling:", hparams.dynamic_loss_scaling)
    print("Distributed Run:", hparams.distributed_run)
    print("cuDNN Enabled:", hparams.cudnn_enabled)
    print("cuDNN Benchmark:", hparams.cudnn_benchmark)



    #train("D:\\OutDir1", "D:\\LogDir1",  args.checkpoint_path,
    #      args.warm_start, args.n_gpus, args.rank, args.group_name, hparams)

    train("D:\\OutDir1", "D:\\LogDir1",  "D:\\OutDir1\\checkpoint_294000",
          False, args.n_gpus, args.rank, args.group_name, hparams)