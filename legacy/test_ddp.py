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
    
    # Find the VPN interface name (likely wg0 for WireGuard)
    vpn_interface = "matcha_t"  # Common name for WireGuard interfaces
    os.environ["NCCL_SOCKET_IFNAME"] = vpn_interface
    
    rank = int(input("Enter rank (0 for server, 1 for client): "))
    world_size = 2
    backend = "nccl"
    
    print(f"Initializing rank {rank} on VPN interface...")
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Server code (rank 0)
    if rank == 0:
        # Create a tensor to send
        tensor_to_send = torch.randn(5, 3, device=device)
        print(f"Server created tensor: {tensor_to_send}")
        
        # Send tensor to client (rank 1)
        dist.send(tensor_to_send, dst=1)
        print("Server sent tensor to client")
        
        # Wait for acknowledgment
        ack = torch.zeros(1, device=device)
        dist.recv(ack, src=1)
        print(f"Server received acknowledgment: {ack.item()}")
    
    else:
        received_tensor = torch.zeros(5, 3, device=device)
        
        # Receive tensor from server (rank 0)
        dist.recv(received_tensor, src=0)
        print(f"Client received tensor: {received_tensor}")
        
        # Send acknowledgment back
        ack = torch.tensor([1.0], device=device)
        dist.send(ack, dst=0)
        print("Client sent acknowledgment")
    
    # Synchronize processes
    dist.barrier()
    print(f"Rank {rank}: Communication completed")

def main():
    # Setup distributed data parallel
    rank = setup_ddp()
    
    # Communicate tensor
    communicate_tensor(rank)
    
    # Cleanup
    dist.destroy_process_group()
    print(f"Rank {rank}: Process group destroyed")

if __name__ == "__main__":
    main()