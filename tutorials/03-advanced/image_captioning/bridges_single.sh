#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU
#SBATCH --ntasks-per-node=32
#SBATCH -t 8:00:00
#SBATCH --gres=gpu:p100:2
#SBATCH --job-name=ic_single_gpu_b8

set -x

cd /pylon5/ac7k4vp/jchoi157/pytorch-tutorial/tutorials/03-advanced/image_captioning

source activate dlc-mpi
export I_MPI_JOB_RESPECT_PROCESS_PLACEMENT=0
#export I_MPI_PIN_DOMAIN=[3fffffff]
export I_MPI_PIN_DOMAIN=[40000000]

#OMP_NUM_THREADS=30 KMP_AFFINITY=granularity=fine,verbose,compact,1,0 KMP_SETTINGS=1 mpiexec -n 1 -genv I_MPI_DEBUG=5 python train_single.py -e 2 --log-step 10 -m -b 8
mpiexec -n 1 -genv I_MPI_DEBUG=5 python train_single.py -e 2 --log-step 10 -m -b 8
