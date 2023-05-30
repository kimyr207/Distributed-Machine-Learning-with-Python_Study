import torch.distributed as dist

# Process group 초기화
dist.init_process_group(backend="nccl")

group = dist.distributed_c10d._get_default_group()
# 새로운 process 그롭 생성
#group = dist.new_group([_ for _ in range(dist.get_world_size())])

print(f"{group} - rank: {dist.get_rank()}\n")