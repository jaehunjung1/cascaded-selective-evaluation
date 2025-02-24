import json
import os
import random
from argparse import ArgumentParser
from pathlib import Path
from typing import List

import jsonlines


def save_to_file(sample_list: List[dict], out_filename: str | Path, save_mode: str = 'w') -> object:
    assert save_mode in ['w', 'a'], "Save mode should be either `w` or `a`."

    if len(sample_list) == 0:
        return

    sample_str_list = [json.dumps(sample) for sample in sample_list]
    with open(out_filename, save_mode) as f:
        f.write("\n".join(sample_str_list) + "\n")


def parse_args():
    parser = ArgumentParser()

    parser.add_argument("--in_filename", default="./data/preprocessed/data.jsonl")

    parser.add_argument("--fewshot_out_filename", default="./data/split/fewshot.jsonl")
    parser.add_argument("--calibration_out_filename", default="./data/split/calibration.jsonl")
    parser.add_argument("--test_out_filename", default="./data/split/test.jsonl")

    parser.add_argument("--N", default=3, help="N for simulated annotators")
    parser.add_argument("--K", default=3, help="K for simulated annotators")

    parser.add_argument("--calibration_set_size", default=500, help="Size of calibration set to split")

    args = parser.parse_args()

    args.fewshot_out_filename = Path(args.fewshot_out_filename)
    args.calibration_out_filename = Path(args.calibration_out_filename)
    args.test_out_filename = Path(args.test_out_filename)

    assert not args.fewshot_out_filename.exists(), f"`{args.fewshot_out_filename}` exists."
    assert not args.calibration_out_filename.exists(), f"`{args.calibration_out_filename}` exists."
    assert not args.test_out_filename.exists(), f"`{args.test_out_filename}` exists."

    return args


if __name__ == "__main__":
    args = parse_args()

    with jsonlines.open(args.in_filename) as f:
        samples = list(f)

    # -- prepare output directory -- #
    os.makedirs("./data/split", exist_ok=True)

    # -- Sample few-shot examples -- #
    # remove trivial samples where either of the response is zero-length
    filtered_samples = []
    for sample in samples:
        output_lengths = [len(output) for output in sample["outputs"]]
        if 0 in output_lengths:
            continue

        filtered_samples.append(sample)

    # sample K annotations per each of N annotators
    fewshot_pool = random.sample(filtered_samples, args.N * args.K)

    # format and save fewshot examples
    fewshot_samples = []
    for annotator_idx in range(args.N):
        evaluated_samples = fewshot_pool[annotator_idx * args.K:(annotator_idx + 1) * args.K]
        fewshot_samples.append({
            "annotator": f"human_{annotator_idx}",
            "evaluated_samples": evaluated_samples,
        })

    save_to_file(fewshot_samples, args.fewshot_out_filename)

    # -- Split calibration and test set -- #
    # split two sets
    random.shuffle(samples)
    calibration_samples, test_samples = samples[:args.calibration_set_size], samples[args.calibration_set_size:]

    # format and save files
    save_to_file(calibration_samples, args.calibration_out_filename)
    save_to_file(test_samples, args.test_out_filename)

# python prepare_data_splits.py --input_filename=./data/preprocessed/data.jsonl --fewshot_out_filename=./data/split/fewshot.jsonl --calibration_out_filename=./data/split/calibration.jsonl --test_out_filename=./data/split/test.jsonl --N=3 --K=3 --calibration_set_size=500










