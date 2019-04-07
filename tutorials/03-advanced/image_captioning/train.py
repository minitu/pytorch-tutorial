print('Importing modules...', flush=True)
import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
import time
from data_loader import get_loader
from build_vocab import Vocabulary
from model import EncoderCNN, DecoderRNN
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
from torch.utils.cpp_extension import load

class Worker:
  def __init__(self, args):
    # Initialize MPI/NCCL and set topology variables
    self.init_dist()
    self.rank = self.dist.get_rank()
    self.world_size = self.dist.get_world_size()
    self.local_rank = self.dist.get_local_rank()
    self.local_size = self.dist.get_local_size()
    self.n_gpus = self.dist.get_n_gpus()
    self.n_nodes = self.world_size / self.local_size
    self.node = self.rank // self.local_size
    self.n_cpu_workers = (self.local_size - self.n_gpus) * self.n_nodes
    self.n_gpu_workers = self.n_gpus * self.n_nodes

    # Calculate batch sizes
    self.total_batch_size = args.batch_size
    self.cpu_batch_size = args.cpu_batch_size
    assert ((self.total_batch_size - self.cpu_batch_size * self.n_cpu_workers * self.n_nodes) \
        % (self.n_gpus * self.n_nodes) == 0), "GPU batch size is not an integer"
    self.gpu_batch_size = int((self.total_batch_size - self.cpu_batch_size * self.n_cpu_workers * self.n_nodes) \
        / (self.n_gpus * self.n_nodes))
    self.batch_size = self.cpu_batch_size if self.dist.is_cpu_rank() else self.gpu_batch_size

    print("[Rank {}] Current CUDA device: {}".format(self.rank, torch.cuda.current_device()), flush=True)

  def init_dist(self):
    # C++ extension module with JIT compilation
    dist_module = load(name="dist", sources=["dist.cu"], verbose=True, with_cuda=True,
        extra_cuda_cflags=['-ccbin', 'g++', '-std=c++11', '-O3',
          #'-I/usr/mpi/gcc/openmpi-2.1.2-hfi/include',
          #'-I/usr/mpi/gcc/mvapich2-2.3b-hfi/include',
          '-I/opt/intel/compilers_and_libraries_2017.4.196/linux/mpi/intel64/include',
          #'-I/opt/intel/compilers_and_libraries_2017.4.196/linux/mpi/include64',
          '-I/pylon5/ac7k4vp/jchoi157/pytorch/build/nccl/include'],
        extra_ldflags=['-L/opt/packages/cuda/9.2/lib64', '-lcudart', '-lrt',
          #'-L/usr/mpi/gcc/openmpi-2.1.2-hfi/lib64', '-lmpi',
          #'-L/usr/mpi/gcc/mvapich2-2.3b-hfi/lib', '-lmpi',
          '-L/opt/intel/compilers_and_libraries_2017.4.196/linux/mpi/intel64/lib', '-lmpi',
          #'-L/opt/intel/compilers_and_libraries_2017.4.196/linux/mpi/lib64', '-lmpi',
          '-L/pylon5/ac7k4vp/jchoi157/pytorch/build/nccl/lib', '-lnccl'],
        build_directory="/home/jchoi157/torch_extensions"
        )
    self.dist = dist_module.DistManager()

  def average_gradients(self):
    # Only all-reduce decoder parameters since encoder is pre-trained
    for param in self.decoder.parameters():
      if self.dist.is_cpu_rank():
        param.grad.data = param.grad.data.cuda(0, non_blocking=True)
        param.grad.data *= (self.cpu_batch_size / self.total_batch_size)
      else:
        param.grad.data *= (self.gpu_batch_size / self.total_batch_size)

      self.dist.hetero_allreduce(param.grad.data)

      if self.dist.is_cpu_rank():
        param.grad.data = param.grad.data.cpu()

  def train(self, args):
    # Create model directory
    if not os.path.exists(args.model_path):
      os.makedirs(args.model_path)

    # Image preprocessing, normalization for the pretrained resnet
    transform = transforms.Compose([
      transforms.RandomCrop(args.crop_size),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      transforms.Normalize((0.485, 0.456, 0.406),
                           (0.229, 0.224, 0.225))])

    # Load vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:
      vocab = pickle.load(f)

    # Build data loader
    data_loader = get_loader(args.image_dir, args.caption_path, vocab,
                             transform, self.rank, self.world_size, self.local_size,
                             self.n_gpus, self.total_batch_size, self.cpu_batch_size,
                             self.gpu_batch_size, self.batch_size, shuffle=True)
    self.num_batches = len(data_loader)
    print("[Rank {}] batch size {}, num batches {}".format(self.rank, self.batch_size,
      self.num_batches), flush=True)

    # Build the models
    self.encoder = EncoderCNN(args.embed_size)
    self.decoder = DecoderRNN(args.embed_size, args.hidden_size, len(vocab), args.num_layers)
    if self.dist.is_gpu_rank():
      self.encoder = self.encoder.cuda(self.local_rank)
      self.decoder = self.decoder.cuda(self.local_rank)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    params = list(self.decoder.parameters()) + list(self.encoder.linear.parameters()) + list(self.encoder.bn.parameters())
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)

    # Train the models
    total_step = len(data_loader)
    for epoch in range(args.num_epochs):
      processed_batches = 0
      batch_time_sum = 0
      batch_start_time = time.time()
      for i, (images, captions, lengths) in enumerate(data_loader):
        # Set mini-batch dataset
        if self.dist.is_gpu_rank():
          images = images.cuda(self.local_rank)
          captions = captions.cuda(self.local_rank)
        targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]

        # Forward, backward, all-reduce and optimize
        features = self.encoder(images)
        outputs = self.decoder(features, captions, lengths)
        loss = criterion(outputs, targets)
        self.decoder.zero_grad()
        self.encoder.zero_grad()
        loss.backward()
        self.average_gradients()
        optimizer.step()

        batch_time = time.time() - batch_start_time
        batch_time_sum += batch_time
        processed_batches += 1

        # Print log info
        if i % args.log_step == 0:
          print('Rank [{}], Epoch [{}/{}], Step [{}/{}], Average time: {:.6f}, Loss: {:.4f}, Perplexity: {:5.4f}'
                  .format(self.rank, epoch, args.num_epochs, i, total_step, batch_time_sum / processed_batches, loss.item(), np.exp(loss.item())), flush=True)
          batch_time_sum = 0
          processed_batches = 0

        # Save the model checkpoints
        if (i+1) % args.save_step == 0:
          torch.save(self.decoder.state_dict(), os.path.join(
            args.model_path, 'decoder-{}-{}.ckpt'.format(epoch+1, i+1)))
          torch.save(self.encoder.state_dict(), os.path.join(
            args.model_path, 'encoder-{}-{}.ckpt'.format(epoch+1, i+1)))

        batch_start_time = time.time()

def main():
  # Training settings
  parser = argparse.ArgumentParser('PyTorch Image Captioning')
  parser.add_argument('--model-path', type=str, default='models/' , help='path for saving trained models')
  parser.add_argument('--crop-size', type=int, default=224 , help='size for randomly cropping images')
  parser.add_argument('--vocab-path', type=str, default='data/vocab.pkl', help='path for vocabulary wrapper')
  parser.add_argument('--image-dir', type=str, default='data/resized2014', help='directory for resized images')
  parser.add_argument('--caption-path', type=str, default='data/annotations/captions_train2014.json', help='path for train annotation json file')
  parser.add_argument('--log-step', type=int , default=10, help='step size for prining log info')
  parser.add_argument('--save-step', type=int , default=1000, help='step size for saving trained models')

  # Model parameters
  parser.add_argument('--embed-size', type=int , default=256, help='dimension of word embedding vectors')
  parser.add_argument('--hidden-size', type=int , default=512, help='dimension of lstm hidden states')
  parser.add_argument('--num-layers', type=int , default=1, help='number of layers in lstm')

  parser.add_argument('-e', '--num-epochs', type=int, default=5)
  parser.add_argument('-b', '--batch-size', type=int, default=128)
  parser.add_argument('-c', '--cpu-batch-size', type=int, default=24)
  parser.add_argument('-l', '--learning-rate', type=float, default=0.001)
  parser.add_argument('-g', '--gpu-only', action='store_true', default=False, help="use only GPUs for training")

  args = parser.parse_args()
  print(args, flush=True)

  # Set RNG seed so that calls to rand() are reproducible
  torch.manual_seed(1234)

  # Create worker abstraction
  worker = Worker(args)
  worker.train(args)

if __name__ == '__main__':
  main()
