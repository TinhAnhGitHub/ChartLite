import torch
import torch.distributed as dist
import os
import socket

def setup_ddp():
    # Use your VPN address as the master address
    os.environ["MASTER_ADDR"] = "10.84.11.29"  # Your VPN address
    os.environ["MASTER_PORT"] = "53154"
    
    # Force IPv4
    os.environ["NCCL_SOCKET_FAMILY"] = "AF_INET"
    os.environ["NCCL_DEBUG"] = "INFO"
    os.environ['TORCH_USE_CUDA_DSA']="1"
    vpn_interface = "matcha_t"  # Common name for WireGuard interfaces
    os.environ["NCCL_SOCKET_IFNAME"] = vpn_interface
    
    # Add NCCL debugging and error handling options
    os.environ["NCCL_BLOCKING_WAIT"] = "1"
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
    os.environ["NCCL_P2P_DISABLE"] = "1"
    
    rank = int(input("Enter rank (0 for server, 1 for client): "))
    world_size = 2
    
    # Use NCCL for GPU, gloo for CPU
    if torch.cuda.is_available():
        backend = "nccl"
    else:
        backend = "gloo"
    
    print(f"Initializing rank {rank} on VPN interface with {backend} backend...")
    dist.init_process_group(
        backend=backend, 
        world_size=world_size, 
        rank=rank,
        init_method=f"tcp://10.84.11.29:53154"  
    )
    print(f"Rank {rank} connected successfully over VPN!")
    return rank

def communicate_tensor(rank):
    # Get the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if rank == 0:
        tensor_to_send = torch.randn(1 ,device=device)
        tensor_to_send = tensor_to_send.contiguous()
        print(f"Server created tensor: {tensor_to_send}")
        
        tensor_to_send = tensor_to_send.contiguous()
        torch.cuda.synchronize()
        dist.send(tensor_to_send, dst=1)
        print("Server sent tensor to client")
        
        ack = torch.zeros(1, device=device)
        dist.recv(ack, src=1)
        print(f"Server received acknowledgment: {ack.item()}")
    
    else:
        received_tensor = torch.zeros(1,device=device).contiguous()
        
        torch.cuda.synchronize()
        dist.recv(received_tensor, src=0)
        print(f"Client received tensor: {received_tensor}")
        
        ack = torch.tensor([1.0], device=device).contiguous()
        dist.send(ack, dst=0)
        print("Client sent acknowledgment")
    
    if device.type == "cuda":
        torch.cuda.synchronize()
        
    dist.barrier()
    print(f"Rank {rank}: Communication completed")

def main():
    try:
        # Setup distributed data parallel
        rank = setup_ddp()
        
        # Communicate tensor
        communicate_tensor(rank)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        if dist.is_initialized():
            dist.destroy_process_group()
            print(f"Rank {rank}: Process group destroyed")


main()