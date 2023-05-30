import torch
import torch.distributed as dist

dist.init_process_group("nccl")
rank = dist.get_rank()
torch.cuda.set_device(rank)

tensor = torch.ones(1).to(torch.cuda.current_device())
# rank 0번에 각 rank의 tensor 를 보내 합친 결과를 rank 0에 저장
dist.reduce(tensor, op=torch.distributed.ReduceOp.SUM, dst=0)

print(f"[{rank}] data = {tensor[0]}")