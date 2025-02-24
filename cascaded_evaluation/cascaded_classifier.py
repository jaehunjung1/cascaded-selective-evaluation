from typing import List, Dict

import ipdb
import numpy as np

from cascaded_evaluation.util import *
from model.openai_model import OpenAIModel
from model.vllm_model import VLLMModel


class CascadedClassifier:
    def __init__(self, model_names: List[str], calibration_samples: Dict[str, List[Dict]], alpha: float, delta: float):
        self.model_names = model_names
        self.models = []  # initialized only when `evaluate_sample` is used

        self.alpha = alpha
        self.delta = delta

        self.lambda_hats = self.calibrate(calibration_samples, alpha, delta)

    def calibrate(self, calibration_samples: Dict[str, List[Dict]], alpha: float, delta: float) -> List[float]:
        """
        set and return `self.lambda_hats` according to `alpha` and `delta`.
        """
        assert all(model_name in calibration_samples for model_name in self.model_names), f"Calibration samples should include {self.model_names}."

        self.alpha = alpha
        self.delta = delta

        # -- prepare data -- #
        merged_samples = merge_data(calibration_samples, self.model_names)
        phats, yhats, labels = prepare_data(merged_samples, self.model_names)

        self.lambda_hats = []

        predicted = torch.zeros_like(labels)
        for name in self.model_names:
            selective_classification_util = SelectiveClassificationUtil(
                cal_phats=phats[name][~predicted.bool()],
                cal_yhats=yhats[name][~predicted.bool()],
                cal_labels=labels[~predicted.bool()],
                delta=delta,
            )

            if selective_classification_util.lambdas.size(-1) == 0:
                lam_hat = self.lambda_hats[-1]
            else:
                lam_hat = selective_classification_util.lambda_hat(alpha=alpha)

            self.lambda_hats.append(lam_hat)
            predicted[phats[name] >= lam_hat] = 1

        return self.lambda_hats

    def apply_decision_rule(self, test_samples: Dict[str, List[Dict]], model_names: List):
        """
        Given test samples where all evaluations are done by each model, apply the decision rule
        """
        assert all(model_name in test_samples for model_name in self.model_names), f"Test samples should include {self.model_names}."

        # -- prepare data -- #
        merged_samples = merge_data(test_samples, self.model_names)
        phats, yhats, labels = prepare_data(merged_samples, self.model_names)

        predictions = torch.ones_like(labels) * -1
        evaluators = torch.ones_like(labels) * -1  # each element is index of the evaluated model
        for idx, name in enumerate(model_names):
            # to evaluate, should be (1) larger than lambda for this model, (2) evaluator should not be already set (i.e. < 0)
            evaluated_predictions = torch.logical_and(phats[name] >= self.lambda_hats[idx], evaluators < 0)

            evaluators[evaluated_predictions] = idx
            predictions[evaluated_predictions] = yhats[name][evaluated_predictions].float()

        if torch.sum(evaluators >= 0) > 0:
            evaluator_composition = {
                name: (torch.sum(evaluators == idx) / evaluators.size(-1)).item()
                for idx, name in enumerate(model_names)
            }
            evaluator_composition = {name: composition / sum(evaluator_composition.values())
                                     for name, composition in evaluator_composition.items()}

            selective_acc = (torch.sum(predictions == labels) / torch.sum(evaluators >= 0)).item()
            coverage = (torch.sum(evaluators >= 0) / evaluators.size(-1)).item()
        else:
            evaluator_composition = {name: 0. for name in model_names}
            selective_acc = 1.0
            coverage = 0.0

        return evaluator_composition, selective_acc, coverage, evaluators

    def initialize_evaluators(self):
        for name in self.model_names:
            if name == "gpt-4-turbo":
                full_name = "gpt-4-turbo-2024-04-09"
            elif name == "gpt-3.5-turbo":
                full_name = "gpt-3.5-turbo"
            elif name == "mistral-7b-instruct":
                full_name = "mistralai/Mistral-7B-Instruct-v0.2"
            else:
                raise NotImplementedError

            if "gpt" in name:
                self.models.append(
                    OpenAIModel(model_name=full_name)
                )
            else:
                self.models.append(
                    VLLMModel(model_name=full_name)
                )

    def evaluate_sample(self, sample: Dict, fewshot_examples_list: List[List[Dict]]):
        """
        evaluate the given samples according to the calibrated thresholds
        """
        if len(self.models) == 0:
            self.initialize_evaluators()

        for idx, (model, model_name) in enumerate(zip(self.models, self.model_names)):
            probs = model.simulate_annotators(sample, fewshot_examples_list)
            if not (probs == [] or np.sum(np.isnan(probs)) > 0):
                if max(probs) >= self.lambda_hats[idx]:
                    return {
                        "probs": probs,
                        "evaluator": model_name,
                    }
        return {
            "probs": None,
            "evaluator": "abstained"
        }

