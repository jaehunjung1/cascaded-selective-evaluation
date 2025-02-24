import jsonlines

from cascaded_evaluation.cascaded_classifier import CascadedClassifier


if __name__ == "__main__":
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

    # -- prepare cascaded classifier -- #
    cascaded_classifier = CascadedClassifier(model_names, calibration_samples=calibration_samples, alpha=0.15, delta=0.1)

    evaluator_composition, selective_acc, coverage, evaluators = cascaded_classifier.apply_decision_rule(test_samples, model_names)
    print(f"Evaluator Composition: {evaluator_composition}")
    print(f"Acc: {selective_acc}")
    print(f"Coverage: {coverage}")




