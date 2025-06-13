import GPUtil
import torch

cuda_available = torch.cuda.is_available()
cuda_version = torch.version.cuda
print(f"CUDA Available: {cuda_available}")
print(f"CUDA Version: {cuda_version}")
print(torch.version.cuda)


pytorch_version = torch.__version__
pytorch_cuda_available = torch.cuda.is_available()
print(f"PyTorch Version: {pytorch_version}")
print(f"PyTorch CUDA Available: {pytorch_cuda_available}")

GPUs = GPUtil.getGPUs()
if len(GPUs) > 0:
    gpu = GPUs[0]
    print(f"Name: {gpu.name}")
    print(f"ID: {gpu.id}")
    print(f"Memory Used: {gpu.memoryUsed}MB")
    print(f"Memory Total: {gpu.memoryTotal}MB")
    print(f"Load: {gpu.load*100}%")
else:
    print("No GPU found")