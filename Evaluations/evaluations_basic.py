import evaluate
import os
import pathlib
import json
import numpy as np
import nltk
import re
import argparse


def read_results(filepath):
    data = []
    with open(filepath, "r", encoding="utf-8") as file:
        raw_lines = file.readlines()
        for raw_line in raw_lines:
            line = json.loads(raw_line)
            data.append(line)

    return data


def calc_total_hallucinations(pred, ref):
    pred_tok = [nltk.word_tokenize(p) for p in pred]
    ref_tok = [[nltk.word_tokenize(p_) for p_ in p] for p in ref]
    ref_tok = [[item for sublist in p for item in sublist] for p in ref_tok]

    source_tok = [[re.sub(r'\W+', '', w).lower() for w in tok] for tok in ref_tok]
    pred_tok = [[re.sub(r'\W+', '', w).lower() for w in tok] for tok in pred_tok]

    hall = []
    for i in range(len(pred_tok)):

        source_set = set(source_tok[i])

        num_halluc = 0
        total_tok = 0

        for word in pred_tok[i]:
            num_halluc += int(word not in source_set)
            total_tok += 1
        try:
            hall.append((1.0 * num_halluc) / (total_tok))
        except:
            continue

    return np.mean(np.array(hall)) * 100


def evaluation_basic(dataset, results_folder, evaluations_folder, mode, models_specified=None):
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

        # calculate basic metrics for the current model
        metrics = ["rouge", "meteor", "bleu"]
        evaluation_results = {}
        for metric in metrics:
            evaluator = evaluate.load(metric)
            evaluation_results[metric] = evaluator.compute(predictions=predictions, references=references)
        evaluation_results["halluc"] = calc_total_hallucinations(pred=predictions, ref=references)
        evaluation_results["avg_length"] = np.mean(np.array([len(nltk.word_tokenize(doc)) for doc in predictions]))
        print(evaluation_results)
        print("\n")

        # save basic metrics for the current model
        output_filename = model + "-basic.json"
        output_filepath = os.path.join(evaluations_folder, output_filename)
        with open(output_filepath, "w", encoding="utf-8") as file:
            line_dict = dict(evaluation_results)
            line_dict["dataset"] = dataset
            line_dict["evaluation"] = "basic"
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

    models_specified = ["Full", "LLaMA", "LLaMAwithRetrieval", "RollingLLaMA", "RollingLLaMAwithRetrieval"]

    for dataset in datasets:
        for backbone in backbones:
            print("\n=== Start Basic Evaluations === " + dataset + " === " + backbone + " ===")

            dataset_folder = "./data/" + dataset
            results_folder = dataset_folder + "/results/" + backbone
            evaluations_folder = dataset_folder + "/evaluations/" + backbone

            if mode == "addition":
                results_folder = dataset_folder + "/results/Others"
                evaluations_folder = dataset_folder + "/evaluations/Others"

            evaluation_basic(
                dataset=dataset,
                results_folder=results_folder,
                evaluations_folder=evaluations_folder,
                mode=mode,
                models_specified=models_specified
            )

            print("=== End Basic Evaluations === " + dataset + " === " + backbone + " ===\n")
