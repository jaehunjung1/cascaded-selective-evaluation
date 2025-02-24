import json
from argparse import ArgumentParser
from pathlib import Path
from typing import List

import jsonlines
import numpy as np
from tqdm import tqdm

from model.openai_model import OpenAIModel


def save_to_file(sample_list: List[dict], out_filename: str | Path, save_mode: str = 'w') -> object:
    assert save_mode in ['w', 'a'], "Save mode should be either `w` or `a`."

    if len(sample_list) == 0:
        return

    sample_str_list = [json.dumps(sample) for sample in sample_list]
    with open(out_filename, save_mode) as f:
        f.write("\n".join(sample_str_list) + "\n")


def parse_args():
    parser = ArgumentParser()

    parser.add_argument("--model_name", type=str, help="OpenAI model name (e.g. gpt-3.5-turbo / gpt-4-turbo)")

    parser.add_argument("--in_filename", type=str, help="Input filename to run evaluator")
    parser.add_argument("--fewshot_in_filename", type=str, help="Input filename for few-shot examples")

    args = parser.parse_args()

    if args.model_name == "gpt-4-turbo":
        args.openai_model_name = "gpt-4-turbo-2024-04-09"
    elif args.model_name == "gpt-3.5-turbo":
        args.openai_model_name = "gpt-3.5-turbo"
    else:
        raise NotImplementedError

    args.in_filename = Path(args.in_filename)
    args.out_filename = Path(f"./result/{args.model_name}.{args.in_filename.stem}.jsonl")

    assert not args.out_filename.exists(), "File already exists."
    print(f"Will be saved to: {args.out_filename}")

    return args


if __name__ == "__main__":
    args = parse_args()

    model = OpenAIModel(model_name=args.openai_model_name)

    with jsonlines.open(args.in_filename) as f:
        samples = list(f)

    # --- prepare few-shot examples --- #
    with jsonlines.open("./data/split/fewshot.jsonl") as f:
        fewshot_examples_list = [sample["evaluated_samples"] for sample in list(f)]

    # --- Simulate Annotators --- #
    for sample in tqdm(samples):
        probs = model.simulate_annotators(sample, fewshot_examples_list)

        if not (probs == [] or np.sum(np.isnan(probs)) > 0):
            sample["probs"] = probs
            save_to_file([sample], args.out_filename, save_mode="a")
