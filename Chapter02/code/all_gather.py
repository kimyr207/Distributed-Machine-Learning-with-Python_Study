import torch
import torch.distributed as dist

dist.init_process_group("nccl")
rank = dist.get_rank()
world_size = 4
torch.cuda.set_device(rank)

tensor = torch.tensor([rank], dtype=torch.float32).to(torch.cuda.current_device())
# 모든 랭크의 tensor를 다른 모든 랭크에 보내고 각각의 랭크는 수신된 tensor를 하나의 리스트로 저장
tensor_list = [torch.empty(1).to(torch.cuda.current_device()) for i in range(world_size)]
dist.all_gather(tensor_list, tensor=tensor )
# 모든 랭크들은 [tensor([0.]), tensor([1.]), tensor([2.]), tensor([3.])] 을 가지게 됨
print(f"[{rank}] data = {tensor_list}")