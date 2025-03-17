import torch
import torch.distributed as dist
import os
import socket
import subprocess

def get_interface_ip(interface_name):
    """Get the IP address of a specific interface"""
    try:
        result = subprocess.check_output(['ip', 'addr', 'show', interface_name]).decode('utf-8')
        print(result)
        for line in result.split('\n'):
            if 'inet ' in line:
                return line.strip().split()[1].split('/')[0]
    except:
        pass
    return None


print(get_interface_ip('matcha_t'))