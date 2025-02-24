import logging
import math
from typing import Dict, List

import numpy as np
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from model.prompts import *


class VLLMModel:
    def __init__(self, model_name: str):
        logging.getLogger("vllm").setLevel(logging.WARNING)

        self.model_name = model_name

        self.result1_converter = {"A": 1, "B": 2}
        self.result2_converter = {"B": 1, "A": 2}

        if any(model in self.model_name for model in ["Mistral-7B-Instruct"]):
            self.llm = LLM(model=model_name, seed=42, dtype="half")
        elif any(model in self.model_name for model in ["Mixtral-8x7B-Instruct"]):
            self.llm = LLM(model=model_name, seed=42, dtype="half", tensor_parallel_size=8)
        else:
            raise NotImplementedError
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def simulate_annotators(self, sample: Dict, fewshot_examples_list: List[List[Dict]]):
        instructions = [sample['instruction']] * len(fewshot_examples_list)
        first_outputs = [sample["outputs"][0]] * len(fewshot_examples_list)
        second_outputs = [sample["outputs"][1]] * len(fewshot_examples_list)

        batch_result1, batch_logprobs1 = self.evaluate_fewshot(
            instructions, assistant_as=first_outputs,
            assistant_bs=second_outputs, fewshot_examples_list=fewshot_examples_list,
        )
        batch_result2, batch_logprobs2 = self.evaluate_fewshot(
            instructions, assistant_as=second_outputs,
            assistant_bs=first_outputs, fewshot_examples_list=fewshot_examples_list,
        )

        predictive_probs = []
        for result1, logprobs1 in zip(batch_result1, batch_logprobs1):
            predictive_probs.append({self.result1_converter[key]: math.exp(value) for key, value in logprobs1.items()})
        for result2, logprobs2 in zip(batch_result2, batch_logprobs2):
            predictive_probs.append({self.result2_converter[key]: math.exp(value) for key, value in logprobs2.items()})

        predictive_probs = [
            np.mean([prob_dict[1] for prob_dict in predictive_probs]),
            np.mean([prob_dict[2] for prob_dict in predictive_probs])
        ]

        return predictive_probs

    def evaluate_fewshot(self, instructions: List[str], assistant_as: List[str],
                         assistant_bs: List[str], fewshot_examples_list: List[List[Dict]]):
        sampling_params = SamplingParams(
            n=1,
            temperature=0,
            max_tokens=5,
            logprobs=5,
        )

        # prompt vllm
        prompt_list = [
            self.prepare_fewshot_prompt(instruction, assistant_a, assistant_b, fewshot_examples)
            for instruction, assistant_a, assistant_b, fewshot_examples
            in zip(instructions, assistant_as, assistant_bs, fewshot_examples_list)
        ]
        output_list = self.llm.generate(prompt_list, sampling_params, use_tqdm=False)

        # post-process outputs
        results, result_log_probs = [], []
        for output in output_list:
            output = output.outputs[0]

            # infer labels
            generation = output.text.strip()
            if generation == "[[A]]":
                result = "A"
            elif generation == "[[B]]":
                result = "B"
            else:
                result = None

            # infer log probs
            if result is not None:
                result_log_prob = {"A": -math.inf, "B": -math.inf}
                token_list = [self.tokenizer.decode(token_id) for token_id in output.token_ids]
                for idx, token in enumerate(token_list):
                    if idx > 0 and "[[" in token_list[idx - 1] and ('A' in token or 'B' in token):
                        top_logprobs = output.logprobs[idx]
                        if any(model in self.model_name for model in ["Mistral-7B-Instruct", "Mixtral-8x7B-Instruct"]):
                            if 28741 in top_logprobs.keys():  # "A"
                                result_log_prob["A"] = top_logprobs[28741].logprob
                            if 28760 in top_logprobs.keys():  # "B"
                                result_log_prob["B"] = top_logprobs[28760].logprob
                            break
                        else:
                            raise NotImplementedError

                if math.isinf(result_log_prob["A"]) and math.isinf(result_log_prob["B"]):
                    # if we have not updated any among "A" and "B", return None
                    result_log_prob = None
            else:
                result_log_prob = None

            if result is not None and result_log_prob is not None:
                results.append(result)
                result_log_probs.append(result_log_prob)

        return results, result_log_probs

    def prepare_fewshot_prompt(self, instruction: str, assistant_a: str, assistant_b: str, fewshot_examples: List[Dict]):
        if any(model in self.model_name for model in ["Mistral-7B-Instruct", "Mixtral-8x7B-Instruct"]):
            prompt = fewshot_inst_prompt

            for example in fewshot_examples:
                preferred_response = "[[A]]" if example["preferences"]["human"] == 1 else "[[B]]"
                prompt += "\n" + fewshot_example_prompt.format(
                    instruction=example["instruction"], assistant_a=example["outputs"][0],
                    assistant_b=example["outputs"][1], preferred_response=preferred_response,
                )

            prompt += "\n" + fewshot_query_prompt.format(instruction=instruction, assistant_a=assistant_a, assistant_b=assistant_b)

            messages = [
                {"role": "user", "content": prompt}
            ]
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)

            return prompt

        else:
            raise NotImplementedError

