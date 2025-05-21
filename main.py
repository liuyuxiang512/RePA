from Model.pipeline import WritingAssistant
import json
import argparse
import logging
import random
import numpy as np
from utils import read_dataset
from pprint import pprint
import os
import pathlib


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default=None, help='config file')
    parser.add_argument('--openai_key', type=str, default=None, help='openai key')
    parser.add_argument('--azure_key', type=str, default=None, help='azure key')
    parser.add_argument('--hf_token', type=str, default=None, help='huggingface access token')
    parser.add_argument('--groq_token', type=str, default=None, help='groq api token')
    parser.add_argument('--replicate_token', type=str, default=None, help="replicate api token")
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    # Read all configurations
    with open(args.config_file, "r") as file:
        configs = json.load(file)
    logging.info('=== config kwargs === ' + json.dumps(configs))

    # configs checking
    assert configs["model"] in ["gpt-4-0125-preview",
                                "meta/meta-llama-3-70b-instruct",
                                "meta-llama/Meta-Llama-3-8B-Instruct",
                                "meta/meta-llama-3-8b-instruct"]
    if "ablation" in configs.keys():
        assert configs["ablation"] in ["Full", "woClarify", "woOutline", "woRefusal", "woRevise"]
    if "baseline" in configs.keys():
        assert configs["baseline"] in ["GPT", "RollingGPT", "GPTwithRetrieval", "RollingGPTwithRetrieval",
                                       "LLaMA", "RollingLLaMA", "LLaMAwithRetrieval", "RollingLLaMAwithRetrieval"]

    # Set seed
    if configs["deterministic"] == "on":
        seed = configs["seed"]
        random.seed(seed)
        np.random.seed(seed)
        logging.info("Process is deterministic based on seed " + str(seed))

    # default args
    retriever_kwargs = {
        "retrievers": configs["retrievers"],
        "topk_Engine": configs["topk_Engine"],
        "exclude_domains": configs["exclude_domains"],
        "BING_SEARCH_V7_SUBSCRIPTION_KEY": args.azure_key,
        # "BING_SEARCH_V7_SUBSCRIPTION_KEY": os.environ["BING_SEARCH_V7_SUBSCRIPTION_KEY"],
        "topk_DPR": configs["topk_DPR"],
        "DPR_knowledge_tensors": configs["DPR_knowledge_tensors"],
        "DPR_knowledge_pieces": configs["DPR_knowledge_pieces"],

        # 'use_ctx': False,
        # 'max_query_length': 16,
        # 'use_full_input_as_query': False,
        # 'use_ctx_for_examplars': False,
    }

    llm_kwargs = {
        "temperature": configs["temperature"],
        "OPENAI_API_KEY": args.openai_key,
        # "OPENAI_API_KEY": os.environ['OPENAI_API_KEY'],
        "n_choices": configs["n_choices"],
        "model": configs["model"],
        "max_generation_tokens": configs["max_generation_tokens"],
        "frequency_penalty": configs["frequency_penalty"],
        "confidence_threshold": configs["confidence_threshold"],
        "num_knowledge_piece": configs["num_knowledge_piece"],
        "outline_error_token": configs["outline_error_token"],
        "calibratedQA_error_token": configs["calibratedQA_error_token"],
        "write_error_token": configs["write_error_token"],
        "hf_access_token": args.hf_token,
        "groq_api_token": args.groq_token,
        "replicate_api_token": args.replicate_token
    }

    dataset = read_dataset(input_file=configs["dataset"])

    if args.debug:
        idx_list = [0]
    else:
        idx_list = range(len(dataset))

    for idx in idx_list:
        assistant = WritingAssistant(
            data_sample=dataset[idx],
            prompts_folder=configs["prompts"],
            llm_kwargs=llm_kwargs,
            retriever_kwargs=retriever_kwargs,
            ablation=configs["ablation"],
            debug=args.debug
        )
        assistant.get_assistant_response()
        data_info = assistant.return_logging_into()
        data_info["dataset_name"] = configs["dataset_name"]
        data_info["idx"] = idx
        data_info["ablation"] = configs["ablation"]

        if args.debug:
            pprint(data_info.keys())
        else:
            output_folder = configs["output_folder"]
            pathlib.Path(output_folder).mkdir(parents=True, exist_ok=True)
            output_filename = str(idx) + "-" + configs["ablation"] + ".json"
            output_filepath = os.path.join(output_folder, output_filename)

            with open(output_filepath, "w", encoding="utf-8") as file:
                json.dump(data_info, file)

            print("\nSave for data " + str(idx) + "\n")
