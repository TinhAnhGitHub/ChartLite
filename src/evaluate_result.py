import os
import json
from tqdm import tqdm
import argparse
import sys
import os

root_dir = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), '..'
    )
)
sys.path.append(root_dir)

from src.utils.constant import TOKEN_MAP
from src.utils.metric_utils_fast import F1ScoreEvaluator, TEDAccuracyEvaluator



def main(data_path_json: str) -> None:

    with open(data_path_json, 'r') as f:
        data_json = json.load(f)

    print(f"{len(data_json)=}")
    predictions = []
    answers = []

    for item in tqdm(data_json):
        
        prediction = item['prediction']
        answer = item['ground_truth']

        predictions.append(prediction)
        answers.append(answer)


    print(f"{len(predictions)}")
    print(f"{len(answers)}")
    f1_score_eval = F1ScoreEvaluator()
    ted_acc_eval = TEDAccuracyEvaluator()

    f1_score = f1_score_eval.evaluate(
        preds=predictions, answers=answers
    )

    ted_acc = ted_acc_eval.evaluate(
        preds=predictions, answers=answers
    )

    print(f"{f1_score=}")
    print(f"{ted_acc=}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_json_path', type=str, required=True, help="Path to the JSON file containing predictions and ground truths")
    args = parser.parse_args()
    main(args.data_json_path)

