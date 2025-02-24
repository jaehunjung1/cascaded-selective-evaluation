import math
import os
import time
from typing import List, Dict, Tuple

import numpy as np
import openai

from openai.types.chat.chat_completion import Choice
from wrapt_timeout_decorator import *

from model.prompts import *


def set_openai_api_key():
    if not (api_key := os.getenv("OPENAI_API_KEY")):
        raise ValueError(f"OPENAI_API_KEY is not set.")

    openai.api_key = api_key
    return api_key


class OpenAIModel:
    def __init__(self, model_name: str):
        self.model_name = model_name

        self.client = openai.Client(api_key=set_openai_api_key())

        self.result1_converter = {"A": 1, "B": 2}
        self.result2_converter = {"B": 1, "A": 2}

    def simulate_annotators(self, sample: Dict, fewshot_examples_list: List[List[Dict]]):
        probs = []
        for fewshot_examples in fewshot_examples_list:
            result1, logprobs1 = self.evaluate_fewshot(
                sample["instruction"], sample["outputs"][0], sample["outputs"][1], fewshot_examples,
            )
            result2, logprobs2 = self.evaluate_fewshot(
                sample["instruction"], sample["outputs"][1], sample["outputs"][0], fewshot_examples,
            )

            if result1 is None or result2 is None:
                continue

            probs1 = {self.result1_converter[key]: math.exp(value) for key, value in logprobs1.items()}
            probs2 = {self.result2_converter[key]: math.exp(value) for key, value in logprobs2.items()}

            probs.extend([probs1, probs2])

        probs = [np.mean([prob_dict[1] for prob_dict in probs]), np.mean([prob_dict[2] for prob_dict in probs])]

        return probs

    def evaluate_fewshot(self, instruction: str, assistant_a: str, assistant_b: str, fewshot_examples: List[Dict]):
        prompt_config = {
            "model": self.model_name,
            "temperature": 0,
            "max_tokens": 10,
            "logprobs": True,
            "top_logprobs": 10,
        }

        system_prompt, prompt = self.prepare_fewshot_prompt(instruction, assistant_a, assistant_b, fewshot_examples)
        output = self.prompt_generation(system_prompt, prompt, prompt_config)

        if output is None:  # BadRequestError
            return None, None

        # infer labels
        generation = output.message.content.strip()
        if "[[A]]" in generation:
            result = "A"
        elif "[[B]]" in generation:
            result = "B"
        else:
            result = None

        # infer log probs
        if result is not None:
            result_log_probs = {"A": -math.inf, "B": -math.inf}
            token_list = [x.token for x in output.logprobs.content]
            for idx, token in enumerate(token_list):
                if idx > 0 and "[[" in token_list[idx - 1] and ('A' in token or 'B' in token):
                    top_logprobs = output.logprobs.content[idx].top_logprobs
                    for top_logprob in top_logprobs:
                        if top_logprob.token in ['A', 'B']:
                            result_log_probs[top_logprob.token] = top_logprob.logprob
                    break
        else:
            result_log_probs = None

        return result, result_log_probs

    @staticmethod
    def prepare_fewshot_prompt(instruction: str, assistant_a: str, assistant_b: str, fewshot_examples: List[Dict]):
        prompt = fewshot_inst_prompt

        for example in fewshot_examples:
            preferred_response = "[[A]]" if example["preferences"]["human"] == 1 else "[[B]]"
            prompt += "\n" + fewshot_example_prompt.format(
                instruction=example["instruction"], assistant_a=example["outputs"][0],
                assistant_b=example["outputs"][1], preferred_response=preferred_response,
            )

        prompt += "\n" + fewshot_query_prompt.format(instruction=instruction, assistant_a=assistant_a, assistant_b=assistant_b)
        return system_prompt, prompt

    @timeout(15, timeout_exception=StopIteration)
    def _request_generation(self, system_prompt: str, prompt: str, prompt_config: Dict) -> Choice:
        completion = self.client.chat.completions.create(
            **prompt_config,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        )

        return completion.choices[0]

    def prompt_generation(self, system_prompt: str, prompt: str, prompt_config: Dict) -> Choice | None:
        try:
            inferred_answer = self._request_generation(system_prompt, prompt, prompt_config)
            return inferred_answer

        except StopIteration as e:
            retry_time = e.retry_after if hasattr(e, 'retry_after') else 1
            print(f"Server Time out. Retrying in {retry_time} seconds...")
            time.sleep(retry_time)
            return self._request_generation(system_prompt, prompt, prompt_config)

        except openai.BadRequestError as e:
            return None

        except openai.OpenAIError as e:
            retry_time = e.retry_after if hasattr(e, 'retry_after') else 1
            print(f"Server Time out. Retrying in {retry_time} seconds...")
            time.sleep(retry_time)
            return self._request_generation(system_prompt, prompt, prompt_config)

        except OSError as e:
            retry_time = e.retry_after if hasattr(e, 'retry_after') else 1
            print(f"Server Time out. Retrying in {retry_time} seconds...")
            time.sleep(retry_time)
            return self._request_generation(system_prompt, prompt, prompt_config)

