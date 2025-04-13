import torch.nn as nn
import torch
from typing import Optional
from timm.models.layers import DropPath



class AdapterPlus(nn.Module):
    """
    Adapter+ module for vision transformers
    Features:
        - Post-Adapter placement
        - Channel-wise scaling
        - Houlsby initialization
        - Stochastic Depth
        - No layer normalization
    """
    def __init__(
        self,
        embed_dim: int,
        bottleneck_dim: int = 8,
        drop_path: float = 0.1,
        dropout:float = 0.0,
        act_layer: nn.Module = nn.GELU,
        bias: bool = True,
        pre_dropout: bool = False
    ) -> None:
        super().__init__()

        self.bottleneck = nn.Sequential(
            nn.Dropout(dropout) if dropout > 0 and pre_dropout else nn.Identity(),
            nn.Linear(embed_dim, bottleneck_dim, bias=bias),
            act_layer() if act_layer else nn.Identity(),
            nn.Dropout() if dropout > 0 else nn.Identity(),
            nn.Linear(bottleneck_dim, embed_dim, bias=bias)
        )
        self.drop_path_a = DropPath(drop_path)
        
        self.scaling = nn.Parameter(torch.ones(embed_dim)) # learnable scaling
        self._initialize_weights(std=0.01)

    def _initialize_weights(self, std: float):
        nn.init.trunc_normal_(
            self.bottleneck[1].weight, std=std, a= -2 * std, b= 2 * std
        )
        if self.bottleneck[1].bias is not None:
            nn.init.zeros_(self.bottleneck[1].bias)
        
        nn.init.trunc_normal_(
            self.bottleneck[4].weight, std=std, a= -2 * std, b= 2 * std
        )
        if self.bottleneck[4].bias is not None:
            nn.init.zeros_(self.bottleneck[4].bias)
        
    def forward(
        self,
        x: torch.Tensor,
        skip: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
            Forward pass through the adapter
            Args:
                x (torch.Tensor): Input tensor of shape (batchsize, seq_len, embed_dim)
                skip (Optional[torch.Tensor]): Skip connection tensor 
            Returns:
                Torch.Tensor: Output tensor
        """

        x = self.drop_path_a(self.bottleneck(x))
        x = x * self.scaling
        if skip is not None:
            x = x + skip
        
        return x


        
        