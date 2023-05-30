import torch
import torch.distributed as dist

dist.init_process_group(backend="nccl")
# 프로세스 그룹 초기화
rank = dist.get_rank()
torch.cuda.set_device(rank)

group = dist.new_group([_ for _ in range(dist.get_world_size())])
# 프로세스 그룹 생성

print(f"{group} - rank: {dist.get_rank()}\n")


def do_reduce(rank: int, size: int):
    # create a group with all processors
    group = dist.new_group(list(range(size)))
    tensor = torch.ones(1)
    # sending all tensors to rank 0 and sum them
    dist.reduce(tensor, dst=0, op=dist.ReduceOp.SUM, group=group)
    # can be dist.ReduceOp.PRODUCT, dist.ReduceOp.MAX, dist.ReduceOp.MIN
    # only rank 0 will have four
    print(f"[{rank}] data = {tensor[0]}")