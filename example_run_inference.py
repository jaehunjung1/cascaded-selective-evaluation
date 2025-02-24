import ipdb
import jsonlines

from cascaded_evaluation.cascaded_classifier import CascadedClassifier


if __name__ == "__main__":
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

    # -- prepare cascaded classifier -- #
    cascaded_classifier = CascadedClassifier(model_names, calibration_samples=calibration_samples, alpha=0.15, delta=0.1)

    # --- prepare few-shot examples --- #
    # in this example, use the fixed provided fewshot examples.
    # alternatively, one may define a pool of held-out samples to dynamically sample few-shot demonstrations.
    with jsonlines.open("./data/split/fewshot.jsonl") as f:
        fewshot_examples_list = [sample["evaluated_samples"] for sample in list(f)]

    result = cascaded_classifier.evaluate_sample(
        sample={
            "instruction": "How many rs are in the word strawberry?",
            "outputs": [
                "There are 3 rs in the word strawberry.",
                "There is no r in the word strawberry.",
            ],
        },
        fewshot_examples_list=fewshot_examples_list,
    )

