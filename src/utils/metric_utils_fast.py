from copy import deepcopy
import json
import os
import random
from collections import defaultdict
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple, Union


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
                

        elif isinstance(data, list):

        else:
            new_data = [str(data).strip()]

        

        

