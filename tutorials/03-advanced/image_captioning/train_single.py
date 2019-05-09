print('Importing modules...', flush=True)
import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
import time
from data_loader_single import get_loader
from build_vocab import Vocabulary
from model import EncoderCNN, DecoderRNN
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
from torch.utils.cpp_extension import load

class Worker:
  def __init__(self, args):
    self.batch_size = args.batch_size
    self.use_gpu = args.use_gpu
    self.use_mpi = args.use_mpi
    if self.use_mpi:
      self.init_dist()

    if self.use_gpu:
      print("Current CUDA device: {}".format(torch.cuda.current_device()), flush=True)
    else:
      num_threads = 30
      #torch.set_num_threads(num_threads)
      print("Setting number of threads: {}".format(num_threads, flush=True))

  def init_dist(self):
    # C++ extension module with JIT compilation
    dist_module = load(name="dist", sources=["dist.cu"], verbose=True, with_cuda=True,
        extra_cuda_cflags=['-ccbin', 'g++', '-std=c++11', '-O3',
          #'-I/usr/mpi/gcc/openmpi-2.1.2-hfi/include',
          #'-I/usr/mpi/gcc/mvapich2-2.3b-hfi/include',
          #'-I/opt/intel/compilers_and_libraries_2017.4.196/linux/mpi/intel64/include',
          '-I/opt/intel/compilers_and_libraries_2018.5.274/linux/mpi/include64',
          '-I/pylon5/ac7k4vp/jchoi157/pytorch-mpi/build/nccl/include'],
        extra_ldflags=['-L/opt/packages/cuda/9.2/lib64', '-lcudart', '-lrt',
          #'-L/usr/mpi/gcc/openmpi-2.1.2-hfi/lib64', '-lmpi',
          #'-L/usr/mpi/gcc/mvapich2-2.3b-hfi/lib', '-lmpi',
          #'-L/opt/intel/compilers_and_libraries_2017.4.196/linux/mpi/intel64/lib', '-lmpi',
          '-L/opt/intel/compilers_and_libraries_2018.5.274/linux/mpi/lib64', '-lmpi',
          '-L/pylon5/ac7k4vp/jchoi157/pytorch-mpi/build/nccl/lib', '-lnccl'],
        build_directory="/home/jchoi157/torch_extensions"
        )
    self.dist = dist_module.DistManager(True, True)

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
                             transform, self.batch_size, shuffle=True)
    self.num_batches = len(data_loader)
    print("Batch size {}, num batches {}".format(self.batch_size, self.num_batches), flush=True)

    # Build the models
    self.encoder = EncoderCNN(args.embed_size)
    self.decoder = DecoderRNN(args.embed_size, args.hidden_size, len(vocab), args.num_layers)
    if self.use_gpu:
      self.encoder = self.encoder.cuda(0)
      self.decoder = self.decoder.cuda(0)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    params = list(self.decoder.parameters()) + list(self.encoder.linear.parameters()) + list(self.encoder.bn.parameters())
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)

    # Train the models
    total_step = len(data_loader)
    for epoch in range(args.num_epochs):
      epoch_start_time = time.time()
      batch_time_sum = 0
      batch_time_total = 0
      processed_batches = 0
      processed_batches_total = 0
      batch_start_time = time.time()
      for i, (images, captions, lengths) in enumerate(data_loader):
        # Set mini-batch dataset
        if self.use_gpu:
          images = images.cuda(0)
          captions = captions.cuda(0)
        targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]

        # Forward, backward, all-reduce and optimize
        features = self.encoder(images)
        outputs = self.decoder(features, captions, lengths)
        loss = criterion(outputs, targets)
        self.decoder.zero_grad()
        self.encoder.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time = time.time() - batch_start_time
        batch_time_sum += batch_time
        batch_time_total += batch_time
        processed_batches += 1
        processed_batches_total += 1

        saved_loss = loss.item()
        # Print log info
        if i % args.log_step == 0 and i != 0:
          print('Epoch [{}/{}], Step [{}/{}], Average time: {:.6f}, Loss: {:.4f}, Perplexity: {:5.4f}'
                  .format(epoch, args.num_epochs, i, total_step, batch_time_sum / processed_batches, loss.item(), np.exp(loss.item())), flush=True)
          batch_time_sum = 0
          processed_batches = 0

        # Save the model checkpoints
        if (i+1) % args.save_step == 0:
          torch.save(self.decoder.state_dict(), os.path.join(
            args.model_path, 'decoder-{}-{}.ckpt'.format(epoch+1, i+1)))
          torch.save(self.encoder.state_dict(), os.path.join(
            args.model_path, 'encoder-{}-{}.ckpt'.format(epoch+1, i+1)))

        batch_start_time = time.time()

      epoch_time = time.time() - epoch_start_time
      print('!!! Epoch [{}], Time: {:.6f}, Average batch time: {:.6f}, Loss: {:.4f}, Perplexity: {:5.4f}'.format(
        epoch, epoch_time, batch_time_total / processed_batches_total, saved_loss, np.exp(saved_loss)), flush=True)

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
  parser.add_argument('-l', '--learning-rate', type=float, default=0.001)
  parser.add_argument('-g', '--use-gpu', action='store_true', default=False, help="use GPU")
  parser.add_argument('-m', '--use-mpi', action='store_true', default=False, help="use MPI")

  args = parser.parse_args()
  print(args, flush=True)

  # Set RNG seed so that calls to rand() are reproducible
  torch.manual_seed(1234)

  # Create worker abstraction
  worker = Worker(args)
  worker.train(args)

if __name__ == '__main__':
  main()
