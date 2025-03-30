import torch
import torch.distributed as dist
import os
import time

def setup_ddp():
    os.environ["MASTER_ADDR"] = "10.84.11.20"  # Your VPN address
    os.environ["MASTER_PORT"] = "53154"
    os.environ["GLOO_SOCKET_FAMILY"] = "AF_INET"
    os.environ["GLOO_SOCKET_IFNAME"] = "matcha_t"
    os.environ["GLOO_DEBUG"] = "INFO"
    
    # WSL-specific settings - try these if still having issues
    os.environ["GLOO_IB_DISABLE"] = "1"
    os.environ["GLOO_P2P_DISABLE"] = "1"
    os.environ["GLOO_BLOCKING_WAIT"] = "1"
    
    rank = int(input("Enter rank (0 for server, 1 for client): "))
    world_size = 2
    
    # Try using gloo backend instead of nccl as a test
    # gloo doesn't use CUDA directly for communication
    backend = "gloo"  # Change to "nccl" if gloo works
    
    print(f"Initializing rank {rank} on VPN interface...")
    dist.init_process_group(
        backend=backend, 
        world_size=world_size, 
        rank=rank
    )
    print(f"Rank {rank} connected successfully over VPN!")
    return rank

def communicate_tensor(rank):
    # Use CPU tensors with gloo backend
    device = torch.device("cpu")
    print(f"Using device: {device}")
    
    if rank == 0:
        # Create a small CPU tensor
        tensor_to_send = torch.randn(2, 2, device=device)
        print(f"Server created tensor: {tensor_to_send}")
        
        # Wait a bit to ensure rank 1 is ready
        time.sleep(2)
        
        # Try non-blocking send
        req = dist.isend(tensor_to_send, dst=1)
        print("Server send initiated")
        
        # Wait for it to complete
        req.wait()
        print("Server send completed")
        
        # Wait for acknowledgment
        ack = torch.zeros(1, device=device)
        dist.recv(ack, src=1)
        print(f"Server received acknowledgment: {ack.item()}")
        
    else:  # rank == 1
        # Small delay to ensure rank 0 has started
        time.sleep(1)
        
        # Create receive tensor
        received_tensor = torch.zeros(2, 2, device=device)
        print(f"Client waiting to receive tensor...")
        
        # Receive tensor
        dist.recv(received_tensor, src=0)
        print(f"Client received tensor: {received_tensor}")
        
        # Send acknowledgment
        ack = torch.tensor([1.0], device=device)
        req = dist.isend(ack, dst=0)
        req.wait()
        print("Client sent acknowledgment")
    
    # Simple barrier
    print(f"Rank {rank}: Waiting at barrier")
    dist.barrier()
    print(f"Rank {rank}: Communication completed")

def main():
    rank = setup_ddp()
    
    try:
        communicate_tensor(rank)
    except Exception as e:
        print(f"Error on rank {rank}: {e}")
    finally:
        # Always cleanup
        dist.destroy_process_group()
        print(f"Rank {rank}: Process group destroyed")

if __name__ == "__main__":
    main()