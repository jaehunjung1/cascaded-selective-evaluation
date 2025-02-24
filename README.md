# Cascaded Selective Evaluation

This is the supplementary code for the paper **Trust or Escalate: LLM Judges with Provable Guarantees for Human Judgement**.

## ***Installation***
```shell
pip install -r requirements.txt
```

## ***Step 1. Preparing LLM Judgements***
**NOTE: We share the pre-generated LLM judgements necessary for running Cascaded Selective Eval in `./result/`. To learn to run the main code, ignore this section and go to Step 2.**
\
\
Prepare the few-shot examples (for simulated annotators), calibration split and test split by running `prepare_data_utils.py`. An example usage is as following:
```shell
python prepare_data_splits.py \
  --input_filename={DATA FILE directory} \
  --fewshot_out_filename={FEWSHOT FILE save directory} \
  --calibration_out_filename={CALIBRATION FILE save directory} \
  --test_out_filename={TEST FILE save directory} \
  --N=3 --K=3 --calibration_set_size=500
```
Next, prompt LLM-as-a-judge on the calibration set by running `evaluate_calibration_*.py`. We provide two versions `evaluate_calibration_openai.py` and `evaluate_calibration_vllm.py`.
An example usage is as following:
```shell
python evaluate_calibration_openai.py \
  --model_name=gpt-4-turbo \
  --in_filename={CALIBRATION FILE directory} \
  --fewshot_in_filename={FEWSHOT FILE directory}
```
Running the script will store generations in `./result/{MODEL NAME}.{CALIBRATION FILENAME}.jsonl`.

For those without API or GPU access, we provide LLM judgements with `gpt-4-turbo`, `gpt-3.5-turbo`, `Mistral-7b-instruct` as judge model in `./result/*.jsonl` files.

## ***Step 2. Cascaded Selective Evaluation***
Cascaded Selective Evaluation is implemented as `CascadedClassifier` class in `cascaded_evaluation` folder.

### Basic Usage
An example script to load model judgements on calibration file, calibrate the abstention policy, and evaluate an unseen sample is in `example_run_inference.py`.

First, load the generations from calibrated samples as follows:
```python
import jsonlines

model_names = ["mistral-7b-instruct", "gpt-3.5-turbo", "gpt-4-turbo"]
calibration_filenames = [
    "./result/mistral-7b-instruct.calibration.jsonl",
    "./result/gpt-3.5-turbo.calibration.jsonl",
    "./result/gpt-4-turbo.calibration.jsonl",
]

calibration_samples = {}
for model_name, input_filename in zip(model_names, calibration_filenames):
    with jsonlines.open(input_filename) as f:
        calibration_samples[model_name] = list(f)
```

Load the cascaded classifier - the classifier will automatically calibrate decision rule upon initialization. In addition, load the provided few-shot examples:
```python
from cascaded_evaluation.cascaded_classifier import CascadedClassifier
cascaded_classifier = CascadedClassifier(model_names, calibration_samples=calibration_samples, alpha=0.15, delta=0.1)

# in this example, use the fixed provided fewshot examples.
# alternatively, one may define a pool of held-out samples to dynamically sample few-shot demonstrations.
with jsonlines.open("./data/split/fewshot.jsonl") as f:
    fewshot_examples_list = [sample["evaluated_samples"] for sample in list(f)]
```

Now the classifier can perform cascaded inference given any unseen example.
```python
result = cascaded_classifier.evaluate_sample(
    sample={
        "instruction": "How many rs are in the word strawberry?",
        "outputs": [
            "There are 3 rs in the word strawberry.",
            "There is no r in the word strawberry.",
        ],
    },
    fewshot_examples_list=fewshot_examples_list,
)  # evaluation done by `gpt-3.5-turbo` in this example
```

### Applying decision rule 
Alternatively, in case one already has full evaluation results across all judge models, `CascadedClassifier` can apply calibrated abstention rules on the model predictions. An example script is provided in `example_apply_decision_rule.py`.

Load predictions for both calibration and test set as follows:
```python
import jsonlines

model_names = ["mistral-7b-instruct", "gpt-3.5-turbo", "gpt-4-turbo"]
calibration_filenames = [
    "./result/mistral-7b-instruct.calibration.jsonl",
    "./result/gpt-3.5-turbo.calibration.jsonl",
    "./result/gpt-4-turbo.calibration.jsonl",
]
test_filenames = [
    "./result/mistral-7b-instruct.test.jsonl",
    "./result/gpt-3.5-turbo.test.jsonl",
    "./result/gpt-4-turbo.test.jsonl",
]

calibration_samples = {}
for model_name, input_filename in zip(model_names, calibration_filenames):
    with jsonlines.open(input_filename) as f:
        calibration_samples[model_name] = list(f)

test_samples = {}
for model_name, input_filename in zip(model_names, test_filenames):
    with jsonlines.open(input_filename) as f:
        test_samples[model_name] = list(f)
```

Load the cascaded classifier:
```python
from cascaded_evaluation.cascaded_classifier import CascadedClassifier

cascaded_classifier = CascadedClassifier(model_names, calibration_samples=calibration_samples, alpha=0.15, delta=0.1)
```

Apply abstention rule to the test predictions and report the metrics.
```python
evaluator_composition, selective_acc, coverage, evaluators = cascaded_classifier.apply_decision_rule(test_samples, model_names)
print(f"Evaluator Composition: {evaluator_composition}")
print(f"Acc: {selective_acc}")
print(f"Coverage: {coverage}")
```

The selective accuracy is around 0.87 in this example, which satisfies the user-prescribed risk of `alpha=0.15`.
