import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def run(rank, size):
    """ Distributed function to be implemented later. """
    print(f"Rank {rank} of {size}")

    tensor = torch.zeros(4, 4)
    if rank == 0:
        tensor = torch.rand(4, 4)
        # Send the tensor to process 1
        dist.send(tensor=tensor, dst=1)
        print(f"Send {tensor}")
    else:
        # Receive tensor from process 0
        dist.recv(tensor=tensor, src=0)
        print(f"Receive {tensor}")

def init_process(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


if __name__ == "__main__":
    size = 2
    processes = []
    mp.set_start_method("spawn")
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
