import json
import os
from utils import get_api_response
import argparse
import pathlib
from pprint import pprint


def read_results(filepath):
    data = []
    with open(filepath, "r", encoding="utf-8") as file:
        raw_lines = file.readlines()
        for raw_line in raw_lines:
            line = json.loads(raw_line)
            data.append(line)

    return data


def construct_context(metric, result, prompts_folder):
    prompt_filepath = os.path.join(prompts_folder, "prompt_" + metric + ".txt")
    with open(prompt_filepath, "r") as file:
        prompt = file.read()

    prompt = prompt.replace("${SOURCE_TEXT}", result["source text"])
    prompt = prompt.replace("${TARGET_TEXT}", result["target text"])
    prompt = prompt.replace("${OUTPUT_TEXT}", result["output text"])

    return prompt


def evaluation_LLM(results_folder, evaluations_folder, prompts_folder, llm_kwargs, mode, models_specified=None):
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

    # run LLM evaluation per model
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

        # prepare output file
        output_filename = model + "-LLM.json"
        output_filepath = os.path.join(evaluations_folder, output_filename)
        with open(output_filepath, "w", encoding="utf-8") as file:
            file.write("")

        # run LLM evaluation per file
        for filename in model_filenames:
            result_filepath = os.path.join(results_folder, filename)
            result = read_results(filepath=result_filepath)
            assert len(result) == 1
            if mode != "addition":
                assert result[0]["ablation"] if "ablation" in result[0].keys() else result[0]["baseline"] == model

            # Run evaluation on the result
            line_dict = dict(result[0])
            metrics = ["single_imitativeness", "single_adaptiveness_reference"]
            for metric in metrics:
                user_text = construct_context(metric=metric, result=result[0], prompts_folder=prompts_folder)
                llm_kwargs["model"] = "meta/meta-llama-3-70b-instruct" if metric == "single_imitativeness" else "gpt-4-0125-preview"
                # try:
                response = get_api_response(content=user_text, llm_kwargs=llm_kwargs)
                # except:
                #     llm_kwargs["model"] = "meta/meta-llama-3-70b-instruct"
                #     response = get_api_response(content=user_text, llm_kwargs=llm_kwargs)
                rating = int(response.split("]]")[0].split("[[")[1])
                explanation = response.split("]]")[1].strip()
                metric = metric.split("_")[1]
                line_dict[metric] = {
                    "rating": rating,
                    "explanation": explanation
                }
            imitativeness = line_dict["imitativeness"]["rating"]
            adaptiveness = line_dict["adaptiveness"]["rating"]
            line_dict["adaptive-imitativeness"] = 2 * imitativeness * adaptiveness / (imitativeness + adaptiveness)

            # save LLM metrics for the current example
            with open(output_filepath, "a", encoding="utf-8") as file:
                json.dump(line_dict, file)
                file.write("\n")


def evaluation_summary(evaluations_folder):
    filenames = os.listdir(evaluations_folder)
    filenames_LLM = []
    for filename in filenames:
        metric = filename.split(".")[0].split("-")[1]
        if metric == "LLM":
            filenames_LLM.append(filename)

    metrics = ["imitativeness", "adaptiveness", "adaptive-imitativeness"]
    model_evaluation_results = {}
    for filename in filenames_LLM:
        model = filename.split(".")[0].split("-")[0]

        evaluation_results = {}
        for metric in metrics:
            evaluation_results[metric] = 0.0

        filepath = os.path.join(evaluations_folder, filename)
        results = read_results(filepath)
        for result in results:
            for metric in metrics:
                try:
                    metric_rating = result[metric]["rating"]
                except:
                    metric_rating = result[metric]

                evaluation_results[metric] += metric_rating

        for metric in metrics:
            evaluation_results[metric] /= len(results)

        model_evaluation_results[model] = evaluation_results

        output_filename = model + "-LLM-summary.json"
        output_filepath = os.path.join(evaluations_folder, output_filename)
        with open(output_filepath, "w", encoding="utf-8") as file:
            json.dump(evaluation_results, file)

    pprint(model_evaluation_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default=None, help='"baseline" or "ablation"')
    parser.add_argument('--openai_key', type=str, default=None, help='openai key')
    parser.add_argument('--replicate_key', type=str, default=None, help='openai key')
    args = parser.parse_args()

    llm_kwargs = {
        "temperature": 0.3,
        "OPENAI_API_KEY": args.openai_key,
        "replicate_api_token": args.replicate_key,
        "n_choices": 1,
        "model": "",
        "max_generation_tokens": 128,
        "frequency_penalty": 0.3,
    }

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
            print("\n=== Start LLM Evaluations === " + dataset + " === " + backbone + " ===")

            dataset_folder = "./data/" + dataset
            results_folder = dataset_folder + "/results/" + backbone
            evaluations_folder = dataset_folder + "/evaluations/" + backbone
            prompts_folder = "./Evaluations/prompts"

            if mode == "addition":
                results_folder = dataset_folder + "/results/Others"
                evaluations_folder = dataset_folder + "/evaluations/Others"

            evaluation_LLM(
                results_folder=results_folder,
                evaluations_folder=evaluations_folder,
                prompts_folder=prompts_folder,
                llm_kwargs=llm_kwargs,
                mode=mode,
                models_specified=models_specified
            )

            evaluation_summary(evaluations_folder=evaluations_folder)

            print("=== End LLM Evaluations === " + dataset + " === " + backbone + " ===\n")
