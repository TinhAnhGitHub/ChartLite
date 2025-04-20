from copy import deepcopy
import json
import os
import random
from collections import defaultdict
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple, Union
from pprint import pprint
from joblib import Parallel, delayed


class BaseJSONEvaluator(ABC):
    @staticmethod
    def flatten(
        data: Dict[Any, Any]
    ) -> List[Tuple[str, Any]]:
        flatten_data = list()
        def _flatten(value, key=""):
            if type(value) is dict:
                for child_key, child_value in value.items():
                    _flatten(child_value, f"{key}.{child_key}" if key else child_key)
            elif type(value) is list:
                for value_item  in value:
                    _flatten(value_item, key)
            else:
                flatten_data.append((key,value))

        _flatten(data)
        return flatten_data

    
    def normalize_dict(
        self,
        data: Union[Dict, List, Any]
    ):
        if not data:
            return {}
        
        if isinstance(data, dict):
            new_data = dict()
            for key in sorted(data.keys(), key=lambda k: (len(k), k)):
                value = self.normalize_dict(data[key])
                if value:
                    if not isinstance(value,list):
                        value=[value]
                    new_data[key]=value

        elif isinstance(data, list):
            if all(isinstance(item,dict) for item in data):
                new_data = []
                for item in data:
                    item = self.normalize_dict(item)
                    if item:
                        new_data.append(item)
            else:
                new_data = [str(item).strip() for item in data]
        else:
            new_data = [str(data).strip()]
        return new_data

    @abstractmethod
    def evaluate_single(
        self, 
        pred: dict,
        gt: dict,
        **kwargs
    ) -> float:
        pass 
    
    def evaluate(
        self, 
        preds: list[dict],
        answers: list[dict],
        n_jobs: int=-1,
        **kwargs
    )->float:
        if not preds or not answers or len(preds) != len(answers):
            raise ValueError("Predictions and answers must be non-empty lists of equal length")
        
        scores = Parallel(n_jobs=n_jobs)(
            delayed(self.evaluate_single)(pred,answers,**kwargs)
            for pred, answer in zip(preds, answers)
        )
        return float(
            sum(scores) / len(scores) if scores else 0.0
        )





        

