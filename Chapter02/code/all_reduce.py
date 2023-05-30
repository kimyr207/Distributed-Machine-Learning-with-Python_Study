import torch
import torch.distributed as dist

dist.init_process_group("nccl")
rank = dist.get_rank()
torch.cuda.set_device(rank)

tensor = torch.ones(1).to(torch.cuda.current_device())
# 각 rank의 tensor를 모두 합친 결과를 다시 모든 rank에 전송
dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

print(f"[{rank}] data = {tensor[0]}")