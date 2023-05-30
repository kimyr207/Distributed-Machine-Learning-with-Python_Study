import torch
import torch.distributed as dist

dist.init_process_group("nccl")
rank = dist.get_rank()
world_size = 4
torch.cuda.set_device(rank)

tensor = torch.tensor([rank], dtype=torch.float32).to(torch.cuda.current_device())
# 모든 랭크의 tensor를 랭크 0번으로 보내고 하나의 리스트에 저장
if rank == 0:
    tensor_list = [torch.empty(1).to(torch.cuda.current_device()) for i in range(world_size)]
    dist.gather(tensor, gather_list=tensor_list, dst=0)
else:
    dist.gather(tensor, gather_list=[], dst=0)
# [tensor([0.]), tensor([1.]), tensor([2.]), tensor([3.])]
if rank == 0:
    print(f"[{rank}] data = {tensor_list}")