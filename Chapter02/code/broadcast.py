import torch
import torch.distributed as dist
dist.init_process_group("nccl")

rank = dist.get_rank()
torch.cuda.set_device(rank)
if rank == 0:
    tensor = torch.tensor([rank], dtype=torch.float32).to(torch.cuda.current_device())
else:
    tensor = torch.empty(1).to(torch.cuda.current_device())
# rank 0번의 tensor ([0.])을 다른 모든 rank에 전송
dist.broadcast(tensor, src=0 )
print(f"[{rank}] data = {tensor}")
