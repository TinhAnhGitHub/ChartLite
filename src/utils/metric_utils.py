from copy import deepcopy
import json
import os
import random
from collections import defaultdict
from typing import Any, Dict, List, Tuple, Union
import torch
import zss
from nltk import edit_distance
from zss import Node
import numpy as np
import pandas as pd
from rapidfuzz.distance.Levenshtein import distance as levenshtein

from nltk import edit_distance


class JSONParseEvaluator:
    """
    A utility class for evaluating JSON parsing performance via two main metrics:

    1. **Normalized Tree Edit Distance (nTED) Accuracy:** 
       - Converts JSON objects to tree representations.
       - Computes the tree edit distance (TED) between predicted and ground truth trees using custom cost functions.
       - Normalizes the TED by dividing by the TED between an empty tree and the ground truth tree.
       - Accuracy is then computed as: accuracy = max(1 - (TED_pred / TED_empty), 0).

    2. **Field-Level F1 Score:**
       - Flattens JSON objects into key-value pairs.
       - Compares predicted fields to ground truth fields.
       - Computes a micro-averaged F1 score based on true positives, false positives, and false negatives.

    Additional utility methods are provided for tree construction and value comparison with numeric and string tolerances.
    """


    @staticmethod
    def flatten(data: Dict[Any, Any]) -> List[Tuple[str, Any]]:
        """
        Flatten a nested dictionary (or list-containing dictionary) into a list of key-value pairs.

        Example:
            Input:
                {
                    "menu": [
                        {"name": ["cake"], "count": ["2"]},
                        {"name": ["juice"], "count": ["1"]},
                    ]
                }
            Output:
                [
                    ("menu.name", "cake"),
                    ("menu.count", "2"),
                    ("menu.name", "juice"),
                    ("menu.count", "1"),
                ]

        Parameters:
            data (Dict[Any, Any]): The JSON-like data to flatten.

        Returns:
            List[Tuple[str, Any]]: A list of flattened key-value pairs.
        """
        flatten_data = list()

        def _flatten(value, key=""):
            if type(value) is dict:
                for child_key, child_value in value.items():
                    _flatten(child_value, f"{key}.{child_key}" if key else child_key)
            elif type(value) is list:
                for value_item in value:
                    _flatten(value_item, key)
            else:
                flatten_data.append((key, value))

        _flatten(data)
        return flatten_data
    


    @staticmethod
    def update_cost(node1: Node, node2: Node) -> int:
        """
        Compute the update cost between two nodes for the tree edit distance calculation.

        - If both nodes are leaf nodes (i.e., contain "<leaf>"), the cost is the string edit distance between their labels (ignoring the "<leaf>" token).
        - If one node is a leaf and the other is not, the cost is 1 plus the length of the non-leaf's label (excluding "<leaf>").
        - If neither node is a leaf, the cost is 0 if the labels are identical, or 1 otherwise.

        Parameters:
            node1 (Node): The first node.
            node2 (Node): The second node.

        Returns:
            int: The cost of updating node1 to node2.
        """

        label1 = node1.label
        label2 = node2.label
        label1_leaf = "<leaf>" in label1
        label2_leaf = "<leaf>" in label2
        if label1_leaf == True and label2_leaf == True:
            return edit_distance(label1.replace("<leaf>", ""), label2.replace("<leaf>", ""))
        elif label1_leaf == False and label2_leaf == True:
            return 1 + len(label2.replace("<leaf>", ""))
        elif label1_leaf == True and label2_leaf == False:
            return 1 + len(label1.replace("<leaf>", ""))
        else:
            return int(label1 != label2)

    @staticmethod
    def insert_and_remove_cost(node: Node):
        """
        Compute the insertion or removal cost for a node during tree edit distance computation.

        - For leaf nodes, the cost is the length of the label (ignoring "<leaf>").
        - For non-leaf nodes, the cost is fixed at 1.

        Parameters:
            node (Node): The node being inserted or removed.

        Returns:
            int: The insertion or removal cost.
        """
        label = node.label
        if "<leaf>" in label:
            return len(label.replace("<leaf>", ""))
        else:
            return 1

    def normalize_dict(self, data: Union[Dict, List, Any]):
        """
        Normalize a JSON-like structure by sorting dictionary keys and ensuring consistency in list formats.

        - For dictionaries: keys are sorted by length and lexicographically.
        - For lists: if the list contains dictionaries, each item is normalized;
          otherwise, non-empty strings/numbers are converted to strings and stripped.
        - Single non-list elements are returned as a single-element list.

        Parameters:
            data (Any): The data to normalize (dict, list, or basic type).

        Returns:
            Any: The normalized data structure.
        """
        if not data:
            return {}

        if isinstance(data, dict):
            new_data = dict()
            for key in sorted(data.keys(), key=lambda k: (len(k), k)):
                value = self.normalize_dict(data[key])
                if value:
                    if not isinstance(value, list):
                        value = [value]
                    new_data[key] = value

        elif isinstance(data, list):
            if all(isinstance(item, dict) for item in data):
                new_data = []
                for item in data:
                    item = self.normalize_dict(item)
                    if item:
                        new_data.append(item)
            else:
                new_data = [str(item).strip() for item in data if type(item) in {str, int, float} and str(item).strip()]
        else:
            new_data = [str(data).strip()]

        return new_data

    def cal_f1(self, preds: List[Dict[Any, Any]], answers: List[Dict[Any, Any]]) -> float:
        """
        Compute the micro-averaged field-level F1 score between predicted and ground truth JSON objects.

        Process:
          1. Normalize and flatten both prediction and ground truth JSONs.
          2. For each predicted key-value pair, check for an exact match (or within tolerance).
          3. Count matching fields as true positives; missing or extra fields contribute as false negatives/positives.
          4. Calculate F1 using the formula: F1 = total_true_positives / (total_true_positives + 0.5 * total_errors).

        Parameters:
            preds (List[Dict[Any, Any]]): List of predicted JSON objects.
            answers (List[Dict[Any, Any]]): List of ground truth JSON objects.

        Returns:
            float: The computed F1 score.
        """
        total_tp, total_fn_or_fp = 0, 0
        for pred, answer in zip(preds, answers):
            pred_flat = self.flatten(self.normalize_dict(pred))
            answer_flat = self.flatten(self.normalize_dict(answer))


            for field in pred_flat:
                if field in answer_flat:
                    total_tp += 1
                    answer_flat.remove(field)
                else:
                    total_fn_or_fp += 1
            total_fn_or_fp += len(answer_flat)
        return total_tp / (total_tp + total_fn_or_fp / 2)
    
    def compute_metric(self,gt: Union[str, float], pred: Union[str, float], numeric_tolerance: float, string_tolerange: int):
        """
        Compare a ground truth value and a predicted value.

        For numeric comparisons:
          - Both values are converted to floats.
          - They are considered a match if the relative error is within `numeric_tolerance`
            (e.g., a tolerance of 0.05 means a 5% allowable difference).

        For string comparisons:
          - The edit distance is computed.
          - They match if the edit distance is less than or equal to `string_tolerance`.

        Parameters:
            gt (Union[str, float]): The ground truth value.
            pred (Union[str, float]): The predicted value.
            numeric_tolerance (float): Relative tolerance for numeric comparison.
            string_tolerance (int): Maximum allowable edit distance for string comparison.

        Returns:
            bool: True if the prediction is within tolerance of the ground truth, otherwise False.
        """
        try:
            # If they are numeric, compute absolute error and check if it's within 5% tolerance
            gt = float(gt)
            pred = float(pred)
            return abs(gt - pred) / abs(gt) <= numeric_tolerance
        except:
            distance = edit_distance(str(gt), str(pred))
            
            # You can define a threshold for acceptable distance
            # For example, if the distance is below a certain threshold, consider it a match
            # This can be adjusted based on tolerance level
            return distance <= string_tolerange

    def compare_json(
        self,
        gt_json: Any,
        pred_json: Any,
        numeric_tolerance: float = 0.05,
        string_tolerance: int = 2
    ) -> Tuple[float, float]:
        """
        Compare two JSON objects (ground truth and prediction) based on their flattened key-value pairs.

        Process:
          1. Normalize and flatten both JSON objects.
          2. For each key in the predicted JSON, check if it exists in the ground truth.
          3. Use `compute_metric` to determine if the values match within the given tolerances.
          4. Record a score of 1 for a match, 0 for a mismatch, and -1 for a missing key.
          5. Calculate:
             - **Overall Metric:** Ratio of correctly matched fields to total fields.
             - **Average Metric:** Mean of the match scores across all fields.

        Parameters:
            gt_json (Any): The ground truth JSON object.
            pred_json (Any): The predicted JSON object.
            numeric_tolerance (float, optional): Tolerance for numeric comparisons. Default is 0.05.
            string_tolerance (int, optional): Tolerance for string comparisons. Default is 2.

        Returns:
            Tuple[float, float]: A tuple containing:
                - overall_metric: The fraction of correctly matched fields.
                - average_metric: The average match score.
        """
        val_metrics: List[Tuple[str, int]] = []
        flat_gt = self.flatten(self.normalize_dict(gt_json))
        flat_pred = self.flatten(self.normalize_dict(pred_json))
        gt_keys = [key for key, _ in flat_gt]

        for index, (pred_key, pred_val) in enumerate(flat_pred):
            if pred_key in gt_keys:
                gt_value = flat_gt[index][1]
                if self.compute_metric(gt_value, pred_val, numeric_tolerance, string_tolerance):
                    val_metrics.append((str(gt_value), 1))
                else:
                    val_metrics.append((str(gt_value), 0))
            else:
                val_metrics.append((str(pred_val), -1))

        total_keys = len(val_metrics)
        correct_matches = sum(1 for _, score in val_metrics if score == 1)
        overall_metric = correct_matches / total_keys if total_keys > 0 else 0.0
        average_metric = sum(score for _, score in val_metrics) / total_keys if total_keys > 0 else 0.0
        return overall_metric, average_metric


    def compare_json_list(
        self,
        gt_jsons: List[Dict[Any, Any]],
        pred_jsons: List[Dict[Any, Any]],
        numeric_tolerance: float = 0.05,
        string_tolerance: int = 2
    ) -> None:
        """
        Compare lists of ground truth and predicted JSON objects by computing metrics for each pair.

        This function iterates through each pair of JSON objects and computes the overall and average
        metrics using the `compare_json` method. It does not return a value but can be extended to
        aggregate or store the metrics.

        Parameters:
            gt_jsons (List[Dict[Any, Any]]): List of ground truth JSON objects.
            pred_jsons (List[Dict[Any, Any]]): List of predicted JSON objects.
            numeric_tolerance (float, optional): Tolerance for numeric comparisons. Default is 0.05.
            string_tolerance (int, optional): Tolerance for string comparisons. Default is 2.
        """

        overall_metrics = []
        average_metrics = []


        for gt_json, pred_json in zip(gt_jsons, pred_jsons):
            overall_metric, average_metric = self.compare_json(gt_json, pred_json, numeric_tolerance, string_tolerance)
            overall_metrics.append(overall_metric)
            average_metrics.append(average_metric)

        return {
            "mean_overall_metric": sum(overall_metrics) / len(overall_metrics) if overall_metrics else 0,
            "mean_average_metric": sum(average_metrics) / len(average_metrics) if average_metrics else 0
        }

    def construct_tree_from_dict(self, data: Union[Dict, List], node_name: str = None):
        """
        Convert Dictionary into Tree

        Example:
            input(dict)

                {
                    "menu": [
                        {"name" : ["cake"], "count" : ["2"]},
                        {"name" : ["juice"], "count" : ["1"]},
                    ]
                }

            output(tree)
                                     <root>
                                       |
                                     menu
                                    /    \
                             <subtree>  <subtree>
                            /      |     |      \
                         name    count  name    count
                        /         |     |         \
                  <leaf>cake  <leaf>2  <leaf>juice  <leaf>1
         """
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
            raise Exception(data, node_name)
        return node


    def cal_acc(self, pred: Dict[Any, Any], answer: Dict[Any, Any]) -> float:
        """
        Calculate the normalized Tree Edit Distance (nTED) based accuracy between a predicted and a ground truth JSON.

        Process:
          1. Normalize and convert both JSON objects into tree representations.
          2. Compute the tree edit distance (TED) between the prediction and the ground truth using custom
             insertion, removal, and update cost functions.
          3. Normalize the distance by comparing it to the TED between an empty tree and the ground truth tree.
          4. Compute accuracy as: accuracy = max(1 - (TED_pred / TED_empty), 0)

        Parameters:
            pred (Dict[Any, Any]): The predicted JSON object.
            answer (Dict[Any, Any]): The ground truth JSON object.

        Returns:
            float: The nTED-based accuracy score (between 0 and 1).
        """
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
