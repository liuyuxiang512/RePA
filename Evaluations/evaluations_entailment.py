import os
import nltk
import pathlib
from transformers import pipeline
import json
from pprint import pprint
import argparse

pipe = pipeline("text-classification", model="geckos/bart-fined-tuned-on-entailment-classification")


def read_results(filepath):
    data = []
    with open(filepath, "r", encoding="utf-8") as file:
        raw_lines = file.readlines()
        for raw_line in raw_lines:
            line = json.loads(raw_line)
            data.append(line)

    return data


def evaluation_entailment(dataset, results_folder, evaluations_folder, mode, models_specified=None):
    # read filenames in results folder
    results_filenames = sorted(os.listdir(results_folder))
    print(len(results_filenames))
    # prepare evaluations folder
    pathlib.Path(evaluations_folder).mkdir(parents=True, exist_ok=True)

    # get model2filenames mappings
    model_filenames_dict = {}
    for filename in results_filenames:
        model = filename.split(".")[0].split("-")[1]
        if model not in model_filenames_dict.keys():
            model_filenames_dict[model] = []
        model_filenames_dict[model].append(filename)

    # run basic evaluation per model
    if mode == "baseline":
        models = list(model_filenames_dict.keys())[:5]
    elif mode == "ablation":
        models = list(model_filenames_dict.keys())[5:]
    else:
        models = model_filenames_dict.keys()

    if models_specified:
        models = models_specified

    for model in models:
        if model not in model_filenames_dict.keys():
            continue
        model_filenames = model_filenames_dict[model]
        print(model + ":" + str(len(model_filenames)))

        # get all predictions and references for the current model
        references = []
        predictions = []
        for filename in model_filenames:
            result_filepath = os.path.join(results_folder, filename)
            result = read_results(filepath=result_filepath)
            assert len(result) == 1
            if mode != "addition":
                assert result[0]["ablation"] if "ablation" in result[0].keys() else result[0]["baseline"] == model
            references.append([result[0]["target text"]])
            predictions.append(result[0]["output text"])

        # calculate entailment metrics for the current model
        labels = ["entailment", "neutral", "contradiction"]
        evaluation_results = {}
        for label in labels:
            evaluation_results[label] = {"mean_count": 0.0, "mean_score": 0.0}
        premise_list = [ref[0] for ref in references]
        hypotheses_list = [nltk.sent_tokenize(pred) for pred in predictions]
        premises_list = [[premise_list[i]] * len(hypotheses_list[i]) for i in range(len(premise_list))]
        for premises, hypotheses in zip(premises_list, hypotheses_list):
            assert len(premises) == len(hypotheses)
            inputs = [premise + "\n" + hypothesis for premise, hypothesis in zip(premises, hypotheses)]
            results = pipe(inputs)
            evaluation_current = {}
            for label in labels:
                evaluation_current[label] = {"count": 0.0, "score": 0.0}
            for result in results:
                label = result["label"]
                score = result["score"]
                evaluation_current[label]["count"] += 1
                evaluation_current[label]["score"] += score
            for label in labels:
                if evaluation_current[label]["count"]:
                    evaluation_results[label]["mean_score"] += evaluation_current[label]["score"] / evaluation_current[label]["count"]
                    evaluation_results[label]["mean_count"] += evaluation_current[label]["count"] / len(results)
        for label in labels:
            evaluation_results[label]["mean_score"] /= len(premises_list)
            evaluation_results[label]["mean_count"] /= len(premises_list)
        pprint(evaluation_results)
        print("\n")

        # save entailment metrics for the current model
        output_filename = model + "-entailment.json"
        output_filepath = os.path.join(evaluations_folder, output_filename)
        with open(output_filepath, "w", encoding="utf-8") as file:
            line_dict = dict(evaluation_results)
            line_dict["dataset"] = dataset
            line_dict["evaluation"] = "entailment"
            json.dump(line_dict, file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default=None, help='"baseline" or "ablation"')
    args = parser.parse_args()
    mode = args.mode

    if mode == "baseline":
        datasets = ["Wikipedia", "RoleEE", "USNews"]
        backbones = ["GPT4", "LLaMA3", "LLaMA8B"]
    elif mode == "ablation":
        datasets = ["Wikipedia"]
        backbones = ["GPT4", "LLaMA3"]
    elif mode == "o1":
        datasets = ["Wikipedia", "RoleEE", "USNews"]
        backbones = ["GPTo1"]
    else:  # addition
        datasets = ["Wikipedia", "RoleEE", "USNews"]
        backbones = ["Default", "Gold", "ETG"]

    print(datasets)
    print(backbones)

    models_specified = ["Full", "LLaMA", "LLaMAwithRetrieval", "RollingLLaMA", "RollingLLaMAwithRetrieval"]

    for dataset in datasets:
        for backbone in backbones:
            print("\n=== Start Entailment Evaluations === " + dataset + " === " + backbone + " ===")

            dataset_folder = "./data/" + dataset
            results_folder = dataset_folder + "/results/" + backbone
            evaluations_folder = dataset_folder + "/evaluations/" + backbone

            if mode == "addition":
                results_folder = dataset_folder + "/results/Others"
                evaluations_folder = dataset_folder + "/evaluations/Others"

            evaluation_entailment(
                dataset=dataset,
                results_folder=results_folder,
                evaluations_folder=evaluations_folder,
                mode=mode,
                models_specified=models_specified
            )

            print("=== End Entailment Evaluations === " + dataset + " === " + backbone + " ===\n")
