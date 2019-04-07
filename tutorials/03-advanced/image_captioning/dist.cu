#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <unistd.h>
#include <stdio.h>

#include "cuda_runtime.h"
#include <nccl.h>
#include "mpi.h"

#define MPI_CHECK(cmd) do {                          \
  int e = cmd;                                      \
  if( e != MPI_SUCCESS ) {                          \
    printf("Failed: MPI error %s:%d '%d'\n",        \
        __FILE__,__LINE__, e);                      \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

#define CUDA_CHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Failed: Cuda error %s:%d '%s'\n",       \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

#define NCCL_CHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",       \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

namespace py = pybind11;

// NCCL type typing
std::map<at::ScalarType, ncclDataType_t> ncclDataType = {
    {at::kChar, ncclInt8},
    {at::kByte, ncclUint8},
    {at::kFloat, ncclFloat},
    {at::kDouble, ncclDouble},
    {at::kInt, ncclInt32},
    {at::kLong, ncclInt64},
    {at::kHalf, ncclHalf},
};

// Helper function that gets the data type and issues error if not supported
ncclDataType_t getNcclDataType(at::ScalarType type) {
  try {
    return ncclDataType.at(type);
  } catch (std::out_of_range& e) {
    throw std::runtime_error("Unsupported data type for NCCL process group");
  }
}

struct DistManager {
  MPI_Comm intra_comm, inter_comm;
  MPI_Group world_group, inter_group;
  ncclUniqueId nccl_id;
  ncclComm_t nccl_comm;
  at::cuda::CUDAEvent nccl_event;
  int rank;
  int world_size;
  int local_rank;
  int local_size;
  int n_nodes;
  int n_gpus;

  DistManager(bool gpu_only) {
    local_rank = 0;
    local_size = 0;

    MPI_CHECK(MPI_Init(NULL, NULL));
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &world_size));

    // Calculate local rank and size based on hostname
    uint64_t hostHashs[world_size];
    char hostname[1024];
    getHostName(hostname, 1024);
    hostHashs[rank] = getHostHash(hostname);
    MPI_CHECK(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hostHashs,
          sizeof(uint64_t), MPI_BYTE, MPI_COMM_WORLD));
    for (int p = 0; p < world_size; p++) {
      if (p == rank) break;
      if (hostHashs[p] == hostHashs[rank]) local_rank++;
    }
    for (int p = 0; p < world_size; p++) {
      if (hostHashs[p] == hostHashs[rank]) local_size++;
    }
    n_nodes = world_size / local_size;

    CUDA_CHECK(cudaGetDeviceCount(&n_gpus));
    if (is_gpu_rank()) {
      CUDA_CHECK(cudaSetDevice(local_rank));
    }
    else {
      CUDA_CHECK(cudaSetDevice(0));
    }

    printf("rank %d, world_size %d, local_rank %d, local_size %d, n_nodes %d, n_gpus %d\n",
        rank, world_size, local_rank, local_size, n_nodes, n_gpus);

    // Create communicators for intra-node and inter-node communication
    if (!gpu_only) {
      MPI_CHECK(MPI_Comm_split(MPI_COMM_WORLD, floor(rank / local_size), rank, &intra_comm));
      MPI_CHECK(MPI_Comm_group(MPI_COMM_WORLD, &world_group));
      int* inter_ranks = (int*)malloc(sizeof(int) * n_nodes);
      for (int i = 0; i < n_nodes; i++)
        inter_ranks[i] = i * local_size + n_gpus;
      MPI_CHECK(MPI_Group_incl(world_group, n_nodes, inter_ranks, &inter_group));
      MPI_CHECK(MPI_Comm_create_group(MPI_COMM_WORLD, inter_group, 0, &inter_comm));
    }

    // Initialize NCCL
    int gpu_rank = ((rank - local_rank) / local_size * n_gpus) + local_rank;
    int n_gpu_ranks = n_nodes * n_gpus;
    if (rank == 0) NCCL_CHECK(ncclGetUniqueId(&nccl_id));
    MPI_CHECK(MPI_Bcast((void*)&nccl_id, sizeof(ncclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD));
    NCCL_CHECK(ncclCommInitRank(&nccl_comm, world_size, nccl_id, rank));

    int nccl_count, nccl_device, nccl_rank;
    NCCL_CHECK(ncclCommCount(nccl_comm, &nccl_count));
    NCCL_CHECK(ncclCommCuDevice(nccl_comm, &nccl_device));
    NCCL_CHECK(ncclCommUserRank(nccl_comm, &nccl_rank));
    printf("[Rank %d] NCCL comm count: %d, device: %d, user rank: %d\n", rank, nccl_count, nccl_device, nccl_rank);

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
  }

  ~DistManager() {
    NCCL_CHECK(ncclCommDestroy(nccl_comm));
    MPI_CHECK(MPI_Finalize());
  }

  void sync_stream(at::Device& device, at::cuda::CUDAEvent& event, at::cuda::CUDAStream& stream) {
    event.record(at::cuda::getCurrentCUDAStream(device.index()));
    event.block(stream);
  }

  // Adapted from ProcessGroupNCCL::allreduce
  void hetero_allreduce(at::Tensor& tensor) {
    /*
    if (is_cpu_rank()) {
      printf("[Rank %d] Before cuda(), tensor.is_cuda(): %d, tensor.device().type(): %d, tensor.device().index(): %d\n", rank, tensor.is_cuda(), tensor.device().type(), tensor.device().index());
      tensor = tensor.to(at::kCUDA);
      printf("[Rank %d] After cuda(), tensor.is_cuda(): %d, tensor.device().type(): %d, tensor.device().index(): %d\n", rank, tensor.is_cuda(), tensor.device().type(), tensor.device().index());
    }
    */

    at::Device device = tensor.device();
    at::cuda::CUDAStream nccl_stream = at::cuda::getStreamFromPool();

    // Wait for device's stream to finish
    sync_stream(device, nccl_event, nccl_stream);

    at::cuda::OptionalCUDAGuard gpuGuard;

    std::unique_lock<std::mutex> cudaFreeMutexLock(
        *(c10::cuda::CUDACachingAllocator::getFreeMutex()));

    NCCL_CHECK(ncclGroupStart());

    gpuGuard.set_index(device.index());

    c10::cuda::CUDACachingAllocator::recordStream(
        tensor.storage().data(), nccl_stream);

    NCCL_CHECK(ncclAllReduce(tensor.data_ptr(), tensor.data_ptr(), tensor.numel(),
          getNcclDataType(tensor.scalar_type()), ncclSum, nccl_comm, nccl_stream.stream()));

    NCCL_CHECK(ncclGroupEnd());

    CUDA_CHECK(cudaStreamSynchronize(nccl_stream.stream()));

    /*
    if (is_cpu_rank()) {
      tensor = tensor.to(at::kCPU);
    }
    */
  }

  static uint64_t getHostHash(const char* string) {
    // Based on DJB2, result = result * 33 + char
    uint64_t result = 5381;
    for (int c = 0; string[c] != '\0'; c++){
      result = ((result << 5) + result) + string[c];
    }
    return result;
  }

  static void getHostName(char* hostname, int maxlen) {
    gethostname(hostname, maxlen);
    for (int i=0; i< maxlen; i++) {
      if (hostname[i] == '.') {
          hostname[i] = '\0';
          return;
      }
    }
  }

  int get_rank() { return rank; }
  int get_world_size() { return world_size; }
  int get_local_rank() { return local_rank; }
  int get_local_size() { return local_size; }
  int get_n_nodes() { return n_nodes; }
  int get_n_gpus() { return n_gpus; }
  bool is_cpu_rank() { return local_rank >= n_gpus; }
  bool is_gpu_rank() { return local_rank < n_gpus; }
};

PYBIND11_MODULE(dist, m) {
  py::class_<DistManager>(m, "DistManager")
    .def(py::init<bool>())
    .def("get_rank", &DistManager::get_rank)
    .def("get_world_size", &DistManager::get_world_size)
    .def("get_local_rank", &DistManager::get_local_rank)
    .def("get_local_size", &DistManager::get_local_size)
    .def("get_n_nodes", &DistManager::get_n_nodes)
    .def("get_n_gpus", &DistManager::get_n_gpus)
    .def("is_cpu_rank", &DistManager::is_cpu_rank)
    .def("is_gpu_rank", &DistManager::is_gpu_rank)
    .def("hetero_allreduce", &DistManager::hetero_allreduce);
}
