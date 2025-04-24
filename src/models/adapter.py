import torch.nn as nn
import torch
from typing import Optional
from timm.models.layers import DropPath
import math


class AdapterPlus(nn.Module):
    def __init__(
        self,
        embed_dim,
        bottleneck_dim=64,
        drop_path=0.1,
        dropout=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        scaling=1.0,
        init="houlsby",
        bias=True,
        pre_dropout=False,
    ):
        super().__init__()
        self.bottleneck = nn.Sequential(
            nn.Dropout(dropout) if dropout > 0 and pre_dropout else nn.Identity(),
            nn.Linear(embed_dim, bottleneck_dim, bias=bias),
            act_layer() if act_layer else nn.Identity(),
            nn.Dropout(dropout) if dropout > 0 and not pre_dropout else nn.Identity(),
            nn.Linear(bottleneck_dim, embed_dim, bias=bias),
        )
        self.norm_a = norm_layer(embed_dim) if norm_layer else nn.Identity()
        self.drop_path_a = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.bottleneck_dim = bottleneck_dim
        if scaling == "learned":
            self.scaling = nn.Parameter(torch.ones(1))
        elif scaling == "channel":
            self.scaling = nn.Parameter(torch.ones(embed_dim))
        else:
            self.scaling = scaling

        if init == "houlsby":
            std = 0.01  
            nn.init.trunc_normal_(
                self.bottleneck[1].weight, std=std, a=-2 * std, b=2 * std
            )
            if self.bottleneck[1].bias is not None:
                nn.init.zeros_(self.bottleneck[1].bias)
            nn.init.trunc_normal_(
                self.bottleneck[4].weight, std=std, a=-2 * std, b=2 * std
            )
            if self.bottleneck[4].bias is not None:
                nn.init.zeros_(self.bottleneck[4].bias)

        elif init == "lora":
            nn.init.kaiming_uniform_(self.bottleneck[1].weight, a=math.sqrt(5))
            if self.bottleneck[1].bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(
                    self.bottleneck[1].weight
                )
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(self.bottleneck[1].bias, -bound, bound)
            nn.init.zeros_(self.bottleneck[4].weight)
            if self.bottleneck[4].bias is not None:
                nn.init.zeros_(self.bottleneck[4].bias)

        elif init == "bert":
            nn.init.normal_(self.bottleneck[1].weight, mean=0.0, std=0.02)
            if self.bottleneck[1].bias is not None:
                nn.init.zeros_(self.bottleneck[1].bias)
            nn.init.normal_(self.bottleneck[4].weight, mean=0.0, std=0.02)
            if self.bottleneck[4].bias is not None:
                nn.init.zeros_(self.bottleneck[4].bias)

        else:
            raise ValueError(f"Initialization {init} not implemented!")

    def forward(
        self, x: torch.Tensor, skip: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = self.norm_a(x)
        x = self.drop_path_a(self.bottleneck(x))
        x = x * self.scaling

        y = x
        if skip is not None:
            y = y + skip

        return y
    

def inject_adapters(encoder_layers, adapter_cls, **adapter_kwargs):
    for i, layer in enumerate(encoder_layers):
        mlp_module = layer.mlp
        if hasattr(mlp_module, 'wo') and isinstance(mlp_module.wo, nn.Linear):
            embed_dim = mlp_module.wo.out_features
        else:
            raise AttributeError(f"Cannot infer embed_dim from MLP layer {i}")

        adapter = adapter_cls(embed_dim=embed_dim, **adapter_kwargs)
        layer.adapter = adapter

        original_layer_forward = layer.forward
        def wrapped_layer_forward(hidden_states, *args, **kwargs):
            outputs = original_layer_forward(hidden_states, *args, **kwargs)

            if isinstance(outputs, torch.Tensor):
                outputs = adapter(outputs, hidden_states)
            elif isinstance(outputs, tuple):
                outputs = (adapter(outputs[0], hidden_states), *outputs[1:])
            return outputs

        layer.forward = wrapped_layer_forward

