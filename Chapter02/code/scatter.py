import torch
import torch.distributed as dist

#NCCL을 scatter operation을 지원하지 않음
dist.init_process_group("gloo")
rank = dist.get_rank()
world_size = 4
torch.cuda.set_device(rank)

tensor = torch.empty(1).to(torch.cuda.current_device())
    # 랭크 0번의 tensor를 world size 만큼 나눠서 각각의 rank에 전송
if rank == 0:
    tensor_list = [torch.tensor([i + 1], dtype=torch.float32).to(torch.cuda.current_device()) for i in range(world_size)]
        # tensor_list = [tensor(1), tensor(2), tensor(3), tensor(4)]
    dist.scatter(tensor, scatter_list=tensor_list, src=0 )
else:
    dist.scatter(tensor, scatter_list=[], src=0 )
print(f"[{rank}] data = {tensor[0]}")