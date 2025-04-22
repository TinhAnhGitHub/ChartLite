from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Union, Optional
import numpy as np
from nltk import edit_distance
import zss
from zss import Node
from joblib import Parallel, delayed
from scipy.optimize import linear_sum_assignment
from copy import deepcopy
from tqdm import tqdm
from nltk import edit_distance
    


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

    @staticmethod
    def normalize_dict(
        data: Union[Dict, List, Any]
    ):
        if not data:
            return {}
        
        if isinstance(data, dict):
            new_data = dict()
            for key in sorted(data.keys(), key=lambda k: (len(k), k)):
                value = BaseJSONEvaluator.normalize_dict(data[key])
                if value:
                    if not isinstance(value,list):
                        value=[value]
                    new_data[key]=value

        elif isinstance(data, list):
            if all(isinstance(item,dict) for item in data):
                new_data = []
                for item in data:
                    item = BaseJSONEvaluator.normalize_dict(item)
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
        answer: dict,
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
            delayed(self.evaluate_single)(pred,answer,**kwargs)
            for pred, answer in tqdm(zip(preds, answers), desc=f"Evaluation of {self.__class__.__name__}....")
        )
        return float(
            sum(scores) / len(scores) if scores else 0.0
        )

    

class F1ScoreEvaluator(BaseJSONEvaluator):
    """
    Compute the field-level F1 Score for a single prediction
    """
    def evaluate_single(
        self,
        pred: dict,
        answer: dict
    ):
        pred_flat = self.flatten(self.normalize_dict(pred))
        answer_flat = self.flatten(self.normalize_dict(answer))

        total_true_positive = 0
        total_false_negative_or_false_positive = 0

        answer_flat_copy = deepcopy(answer_flat)

        for field in pred_flat:
            if field in answer_flat_copy:
                total_true_positive += 1
                answer_flat_copy.remove(field)
            else:
                total_false_negative_or_false_positive += 1
        
        total_false_negative_or_false_positive += len(answer_flat_copy)

        return total_true_positive / (total_true_positive + total_false_negative_or_false_positive / 2) if  (total_true_positive + total_false_negative_or_false_positive / 2) > 0 else 0.0


class TEDAccuracyEvaluator(BaseJSONEvaluator):
    @staticmethod
    def update_cost(node1: Node, node2: Node) -> int:
        """
        Compute the update cost between two nodes for the tree edit distance calculation.
        """
        label1 = node1.label
        label2 = node2.label
        label1_leaf = "<leaf>" in label1
        label2_leaf = "<leaf>" in label2
        
        if label1_leaf and label2_leaf:
            return edit_distance(label1.replace("<leaf>", ""), label2.replace("<leaf>", ""))
        elif not label1_leaf and label2_leaf:
            return 1 + len(label2.replace("<leaf>", ""))
        elif label1_leaf and not label2_leaf:
            return 1 + len(label1.replace("<leaf>", ""))
        else:
            return int(label1 != label2)

    @staticmethod
    def insert_and_remove_cost(node: Node) -> int:
        """
        Compute the insertion or removal cost for a node during tree edit distance computation.
        """
        label = node.label
        if "<leaf>" in label:
            return len(label.replace("<leaf>", ""))
        else:
            return 1

    def construct_tree_from_dict(self, data: Union[Dict, List], node_name: str = None) -> Node:
        if node_name is None:
            node_name = "<root>"

        node = Node(node_name)

        if isinstance(data, dict):
            for key, value in data.items():
                kid_node = self.construct_tree_from_dict(value, key)
                node.addkid(kid_node)
        elif isinstance(data, list):
            if all(isinstance(item, dict) for item in data):
                for item in data:
                    kid_node = self.construct_tree_from_dict(
                        item,
                        "<subtree>",
                    )
                    node.addkid(kid_node)
            else:
                for item in data:
                    node.addkid(Node(f"<leaf>{item}"))
        else:
            if isinstance(data, (str, int, float, bool)):
                return Node(f"<leaf>{data}")
            else:
                return node
        return node
    


    def evaluate_single(
        self,
        pred,
        answer
    )->float:
        pred_tree = self.construct_tree_from_dict(self.normalize_dict(pred))
        answer_tree = self.construct_tree_from_dict(self.normalize_dict(answer))

        ted_pred = zss.distance(
            pred_tree,
            answer_tree,
            get_children=zss.Node.get_children,
            insert_cost=self.insert_and_remove_cost,
            remove_cost=self.insert_and_remove_cost,
            update_cost=self.update_cost,
            return_operations=False,
        )

        empty_tree = self.construct_tree_from_dict(self.normalize_dict({}))
        ted_empty = zss.distance(
            empty_tree,
            answer_tree,
            get_children=zss.Node.get_children,
            insert_cost=self.insert_and_remove_cost,
            remove_cost=self.insert_and_remove_cost,
            update_cost=self.update_cost,
            return_operations=False,
        )

        normalized_distance = ted_pred / ted_empty if ted_empty != 0 else 0
        accuracy = max(1 - normalized_distance, 0)
        return accuracy





class ContinuousEvaluator(BaseJSONEvaluator):
    """
    Evaluator for time-series curve ( for line plot)
    """
    @staticmethod
    def _intervals(xs: np.ndarray):
        """
        Compute intervals based on the following equation
        interval(i, G)  = 
            - (u_i+1 - u_i) / 2
            - (u_i - i_i-1) / 2
            - (u_(i+1)- u_(i-1)) / 2 
        """
        diffs = np.diff(xs)
        intervals = np.empty_like(xs)
        intervals[0] = diffs[0] / 2.0
        intervals[-1] = diffs[-1] / 2.0
        intervals[1:-1] = (xs[2:] - xs[:-2]) / 2.0
        return intervals
    

    @staticmethod
    def _interp_extrapolate(
        x: np.ndarray,
        xp: np.ndarray,
        yp: np.ndarray  
    ):
        
        order = np.argsort(xp)
        xp_s, yp_s = xp[order], yp[order]
        slopes = np.diff(yp_s) / np.diff(xp_s)

        y = np.empty_like(x, dtype=float)
        mask_mid = (x >= xp_s[0]) & (x <= xp_s[-1])
        if mask_mid.any():
            y[mask_mid] = np.interp(x[mask_mid], xp_s, yp_s)
        mask_left = x < xp_s[0]
        if mask_left.any():
            y[mask_left] = yp_s[0] + slopes[0] * (x[mask_left] - xp_s[0])
        mask_right = x > xp_s[-1]
        if mask_right.any():
            y[mask_right] = yp_s[-1] + slopes[-1] * (x[mask_right] - xp_s[-1])
        return y
    

    @staticmethod
    def _errors(
        gt_x: np.ndarray,
        gt_y: np.ndarray,
        pred_x: np.ndarray,
        pred_y: np.ndarray,
        eps: float
    ):
        interp_vals = ContinuousEvaluator._interp_extrapolate(gt_x, pred_x, pred_y)
        errs = np.abs(gt_y - interp_vals) / (gt_y + eps)
        return np.minimum(errs, 1.0)
    
    @staticmethod
    def recall(
        pred_data: np.ndarray,
        gt_data: np.ndarray,
        epsilon: Optional[float] = None
    ):
        gt_x, gt_y = gt_data[:, 0], gt_data[:, 1]
        pred_x, pred_y = pred_data[:, 0], pred_data[:, 1]
        if epsilon is None:
            epsilon = (gt_y.max() - gt_y.min()) / 100.0
        intervals = ContinuousEvaluator._intervals(gt_x)
        errs = ContinuousEvaluator._errors(gt_x, gt_y, pred_x, pred_y, epsilon)
        recall_raw = np.sum((1.0 - errs) * intervals)
        return recall_raw / (gt_x[-1] - gt_x[0])


    def evaluate_single(self, pred, answer):
        """
        Evaluate dataseries of a single chart, and this chart must be line plot kind
        """
        gt_data_series = answer['data_series']
        pred_data_series = pred.get('data_series', None)
        if pred_data_series is None:
            return 0

        try:
            gt_data_series = [
                [float(item[0]), float(item[1])] for item in gt_data_series
            ]
            pred_data_series = [
                [float(item[0]), float(item[1])] for item in pred_data_series
            ]
            pred_data_series_np = np.array(pred_data_series)
            gt_data_series_np = np.array(gt_data_series)

            recall = ContinuousEvaluator.recall(pred_data_series_np, gt_data_series_np)
            return recall
        except Exception:
            print(f"Prediction or ground truth x or y axis cannot be parse to float")
            return 0.0

    



class PointSetEvaluator(BaseJSONEvaluator):
    @staticmethod
    def _compute(
        pred_pts: np.ndarray,
        gt_pts: np.ndarray,
        lambda_scale: float = 1.0
    ):

        cov = np.cov(gt_pts, rowvar=False)
        inv_conv = np.linalg.pinv(cov)

        meansq = np.mean(gt_pts, axis=0) ** 2
        max_diag = 400.0 / meansq

        np.fill_diagonal(
            inv_conv, np.minimum(np.diag(inv_conv), max_diag)
        )

        #diff[n,m,:] = gt_pts[m] - pred_pts[n]
        diff = gt_pts[None, :, :]  - pred_pts[:, None, :]
        d2 = np.einsum(
            "nkd,dd,nkd->nk", diff, inv_conv, diff
        )
        C = np.minimum(1.0, d2**lambda_scale)

        N, M = C.shape
        K = max(N, M)
        if N != M:
            pad = np.ones((K, K), dtype=float)
            pad[:N, :M] = C
            C = pad
        
        rows, cols = linear_sum_assignment(C)
        cost = C[rows, cols].sum()
        return 1.0 - cost/K
    



# class DiscreteEvaluator(BaseJSONEvaluator):
#     def __init__(self, alpha: float=0.5, sigma2: float = 1.0):
#         self.alpha = alpha
#         self.sigma2 = sigma2
    
#     @staticmethod
#     def _norm_edit_distance(a: str, b: str) -> float:
#         d = edit_distance(a,b)
#         m = min(len(a), len(b))
#         if m - d <= 0:


#     def _pairwise_dist(
#         self, 
#         x: str,
#         y_str: Union[str,float],
#         u: str,
#         v_str: Union[str, float],
#     ) -> float:
#         try:
#             y = float(str(y_str).rstrip('%'))
#         except:
#             y = 0.0
#         try:
#             v = float(str(v_str).rstrip('%'))
#         except:
#             v = 0.0
        
#         L_norm = self._norm_edit_distance(x, u)
#         sim = 1.0 - L_norm

#         frac = abs(y-v) / np.sqrt()
        




if __name__ == "__main__":
    sample_dict =  {
            "chart_type": "vbar",
            "plot_bb": {
                "x0": '65',
                "y0": '41',
                "x2": '464',
                "y2": '347'
            },
            "data_series": [
                {"x": '2017', "y": "74.2%", "color": "lapis"},
                {"x": '2018', "y": "73%", "color": "lapis"},
                {"x": '2017', "y": "25.8%", "color": "vampire_black"},
                {"x": '2018', "y": "27%", "color": "vampire_black"}
            ],
            "text_display": [
                {
                    "polygon": {"x0": '157', "y0": '356', "x2": '374', "y2": '367'},
                    "text": ["2017","2018"],
                    "role": "x_axis",
                    "colors": 'None'
                },
                {
                    "polygon": {"x0": '25', "y0": '150', "x2": '374', "y2": '367'},
                    "text": "Share of internet sales",
                    "role": "y_title",
                    "colors": 'None'
                },
                {
                    "polygon": {"x0": '41', "y0": '27', "x2": '59', "y2": '366'},
                    "text": ["0", "1", "2", "3", "4", "5"],
                    "role": "y_axis",
                    "colors": 'None'
                },
                {
                    "polygon": {"x0": '160', "y0": '382', "x2": '325', "y2": '413'},
                    "text": "North America Rest of the world",
                    "role": "legend",
                    "colors": ["lapis", "vampire_black"]
                }
            ]
        }
    sample_dict_2 =  {
                "chart_type": "vbar",
                "plot_bb": {
                    "x0": '65',
                    "y0": '41',
                    "x2": '464',
                    "y2": '347'
                },
                "data_series": [
                    {"x": '2017', "y": "74.2%", "color": "lapis"},
                    {"x": '2018', "y": "73%", "color": "lapis"},
                    {"x": '2017', "y": "25.8%", "color": "vampire_black"},
                    {"x": '2018', "y": "27%", "color": "vampire_black"}
                ],
                "text_display": [
                    {
                        "polygon": {"x0": '157', "y0": '356', "x2": '374', "y2": '367'},
                        "text": ["2017","2018"],
                        "role": "x_axis",
                        "colors": 'None'
                    },
                    {
                        "polygon": {"x0": '25', "y0": '150', "x2": '374', "y2": '367'},
                        "text": "Share of internet sales",
                        "role": "y_title",
                        "colors": 'None'
                    },
                    {
                        "polygon": {"x0": '41', "y0": '27', "x2": '59', "y2": '366'},
                        "text": ["0", "1", "2", "3", "4", "5"],
                        "role": "y_axis",
                        "colors": 'None'
                    },
                    {
                        "polygon": {"x0": '160', "y0": '382', "x2": '325', "y2": '413'},
                        "text": "North America Rest of the world",
                        "role": "legend",
                        "colors": ["lapis", "vampire_black"]
                    }
                ]
            }
    json_e = F1ScoreEvaluator()
    ted = TEDAccuracyEvaluator()

    print(json_e.evaluate(
        [sample_dict],
        [sample_dict],
    ))
    print(ted.evaluate(
        [sample_dict],
        [sample_dict],
    ))
    # json_e = TEDAccuracyEvaluator()
    # print(
    #     json_e.construct_tree_from_dict(sample_dict)
    # )
