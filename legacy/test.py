import torch

# 1. Allocate a block of memory on the CPU
cpu_ram = torch.randn(5, 3)  # Creating a random tensor on CPU (5x3 matrix)
print(f"Original CPU RAM: \n{cpu_ram}")

# 2. Turn it into a block on the device (GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpu_ram = cpu_ram.to(device)  # Move tensor to GPU
print(f"GPU RAM: \n{gpu_ram}")

# 3. Copy the result back from the GPU to the local CPU
cpu_result = gpu_ram.cpu()  # Move tensor back to CPU
print(f"Result copied back to CPU: \n{cpu_result}")