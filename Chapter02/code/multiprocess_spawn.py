import os
import torch.distributed as dist
import torch.multiprocessing as mp

# 각 subprocess에서 실행될 내용
def worker(rank, world_size):
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    group = dist.distributed_c10d._get_default_group()
# 원하는 랭크만 새로운 그룹으로 묶기
#    group = dist.new_group([_ for _ in range(world_size)])
#    group = dist.new_group([0, 1])
    print(f"{group} - rank: {rank}")

# Mainprocess에서 실행될 내용
if __name__ == '__main__':
    world_size = 4
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "32900"
    os.environ["WORLD_SIZE"] = str(world_size)
    
    mp.spawn(worker, args=(world_size,), nprocs=world_size, start_method="spawn", )