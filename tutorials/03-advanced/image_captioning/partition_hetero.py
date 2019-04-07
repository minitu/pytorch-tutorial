from random import Random

class Partition(object):
  def __init__(self, data, index):
    self.data = data
    self.index = index

  def __len__(self):
    return len(self.index)

  def __getitem__(self, index):
    data_idx = self.index[index]
    return self.data[data_idx]

class DataPartitioner(object):
  def __init__(self, data, rank, world_size, local_size, n_gpus, total_batch, cpu_batch, gpu_batch, seed=1234):
    self.data = data
    data_len = len(data)
    num_batches = data_len // total_batch
    used_len = num_batches * total_batch

    if rank == 0:
      print("Using {} samples out of {}".format(used_len, data_len))

    rng = Random()
    rng.seed(seed)
    indexes = [x for x in range(0, used_len)]
    rng.shuffle(indexes)
    self.partitions = []

    for i in range(world_size):
      local_rank = i % local_size
      batch = gpu_batch if local_rank < n_gpus else cpu_batch
      part_len = int(batch * num_batches)
      self.partitions.append(indexes[0:part_len])
      indexes = indexes[part_len:]

  def use(self, partition):
    return Partition(self.data, self.partitions[partition])
