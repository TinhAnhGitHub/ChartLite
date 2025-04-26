from abc import ABC, abstractmethod
from typing import Dict, Any
import torch


class BaseChartExtractionModel(ABC):
    """
    Abstract base class for data extraction models
    """
    def __init__(self):
        pass

    @abstractmethod
    def preprocess(self, image: Any, text: str = None) -> Dict[str, torch.Tensor]:
        """
        Turn raw image and optional text into model output tensors
        Return a dict of tensor
        """
        pass

    @abstractmethod
    def forward(self, batch: Dict[str, torch.Tensor]) -> Any:
        pass

    @abstractmethod
    def generate(self, batch: Dict[str, torch.Tensor], **gen_kwargs) -> Any:
        """
        Generate predictions from the model (inference)
        """

        pass



def build_model(cfg) -> BaseChartExtractionModel:
    model_type = cfg.model.type.lower()
    if model_type == 'matcha':
        from src.models.matcha.matcha_model import Matcha
        return Matcha(cfg)
    elif model_type == 'deplot':
        from src.models.deplot.deplot_model import Deplot
        return Deplot(cfg)
    elif model_type == "pix2struct":
        from src.models.pix2struct.pix2struct_model import Pix2Struct
        return Pix2Struct(cfg)
    elif model_type == 'unichart':
        from src.models.unichart.unichart_model import UniChart
        return UniChart(cfg)
    else:
        raise ValueError(f"Unknown model type: {cfg.model.type}")

    
    

