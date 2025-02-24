from collections import Counter
from typing import Dict, List, Tuple

import torch
from scipy.optimize import brentq
from scipy.stats import binom


def merge_data(samples: Dict[str, List[Dict]], model_names: List[str]) -> List[Dict]:
    """
    Merge list of outputs from different models into a single list
    """
    # -- prepare `unique_sample_dict`: list of unique samples -- #
    unique_sample_dict = {}
    for model_name in model_names:
        for sample in samples[model_name]:
            sample_key = f"{sample['instruction']}-{sample['outputs']}"

            if sample_key in unique_sample_dict:
                unique_sample_dict[sample_key]["probs"][model_name] = sample["probs"]
            else:
                unique_sample_dict[sample_key] = {
                    "outputs": sample["outputs"],
                    "instruction": sample["instruction"],
                    "preferences": sample["preferences"],
                    "probs": {model_name: sample["probs"]}
                }

    # -- filter samples where probs exist for all models -- #
    merged_samples = []
    for sample in unique_sample_dict.values():
        if len(sample["probs"]) == len(model_names):
            merged_samples.append(sample)

    return merged_samples


def prepare_data(merged_samples: List[Dict], model_names: List[str]) -> Tuple[Dict, Dict, torch.Tensor]:
    """
    Transform `merged_samples` into tensors
    Output:
        phats: Dictionary of predicted probability - e.g. {'mistral-7b-instruct': [[0.8, 0.2, 0.3, ...]}
        yhats: Dictionary of predicted label - e.g. {'mistral-7b-instruct': [1, 0, 0, ...]}
        labels: ground-truth label - e.g. [1, 0, 1, ...]
    """
    labels = []
    # Create `predicted_probs`
    # e.g. {"mistral-7b-instruct": [[0.4, 0.6], ...], "gpt-3.5": [[0.2, 0.8], ...], "gpt-4": [[0.25, 0.75], ...]}
    predicted_probs = {model_name: [] for model_name in model_names}
    for sample in merged_samples:
        human_labels = list(sample["preferences"].values())
        labels.append(Counter(human_labels).most_common(1)[0][0])

        for model_name in model_names:
            predicted_probs[model_name].append(sample["probs"][model_name])

    # Labels in the example files are: "1": A is better, "2": B is better
    labels = torch.Tensor(labels) - 1

    phats, yhats = {}, {}
    for model_name, probs in predicted_probs.items():
        probs = torch.Tensor(probs)
        phats[model_name], yhats[model_name] = torch.max(probs, dim=-1)

    return phats, yhats, labels


class SelectiveClassificationUtil:
    def __init__(self, cal_phats: torch.Tensor, cal_yhats: torch.Tensor, cal_labels: torch.Tensor, delta: float):
        self.cal_phats = cal_phats
        self.cal_yhats = cal_yhats
        self.cal_labels = cal_labels

        # tolerance
        self.delta = delta

        # candidate lambdas: make sure there's data for top bin
        self.lambdas = torch.Tensor([lam for lam in torch.linspace(0, 1, 5000) if self.n_lambda(lam) >= 30])

    def selective_risk(self, lam: float):
        # compute empirical risk \hat{R}(\lambda) on calibration set
        num_correct = (self.cal_yhats[self.cal_phats >= lam] != self.cal_labels[self.cal_phats >= lam]).sum()
        num_total = (self.cal_phats >= lam).sum()

        return num_correct / num_total

    def selective_risk_upper_bound(self, lam: float):
        # compute upper bound for empirical risk \hat{R}+(\lambda)

        # upper-bound condition for supremum
        def _upper_bound_condition(r: float, lam: float):
            return binom.cdf(self.selective_risk(lam) * self.n_lambda(lam), self.n_lambda(lam), r) - self.delta

        return brentq(_upper_bound_condition, 0, 0.9999, args=(lam,))

    def n_lambda(self, lam: float):
        return (self.cal_phats >= lam).sum()

    def lambda_hat(self, alpha: float):
        if len(self.lambdas) == 0:
            return 0.

        # prevents null-value
        if self.selective_risk_upper_bound(self.lambdas[-1]) > alpha:
            lam_hat = self.lambdas[-1]
            return lam_hat

        # test from the highest lambda, and stop when infimum condition is met
        for lam in torch.flip(self.lambdas, dims=(-1,)):
            next_lam = lam - 1 / self.lambdas.size(-1)
            if self.selective_risk_upper_bound(next_lam) > alpha:
                lam_hat = lam
                return lam_hat

        # if even the smallest lambda satisfies the condition, return the smallest lambda
        return self.lambdas[0]



