import json
import os

import nltk
import wikipedia
from wikipedia2vec import Wikipedia2Vec
import pathlib
import argparse

from transformers import AutoModel
from numpy.linalg import norm
import numpy as np

from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
import torch

import random

from utils_data import mix_topic_pairs, construct_knowledge_source, save_knowledge_tensors, obtain_knowledge_tensors

embedding_file = "/home/yuxiang/wikipedia/enwiki-20240401-embeddings"
print("... Loading pre-trained embeddings ...")
wiki2vec = Wikipedia2Vec.load(embedding_file)

division = 100
device = "cuda:0"


def process_satellite(input_folder, output_folder, category):
    input_folder_path = os.path.join(input_folder, category)
    output_folder_path = os.path.join(output_folder, category)
    pathlib.Path(output_folder_path).mkdir(parents=True, exist_ok=True)
    for f in os.listdir(output_folder_path):
        os.remove(os.path.join(output_folder_path, f))

    # Step 1
    file_names = os.listdir(input_folder_path)
    print(len(file_names))
    for i, file_name in enumerate(file_names):
        title = ".".join(file_name.split(".")[:-1])
        num = int(title.split("-")[1])
        if 35 <= num <= 63:
            idx = 0
        elif 145 <= num <= 201:
            idx = 1
        else:
            idx = 2
        wiki_page = wikipedia.page(title, auto_suggest=False)
        wiki_title = wiki_page.title
        wiki_summary = wiki_page.content.split("\n")[0].strip()
        line_dict = {
            "title": wiki_title,
            "summary": wiki_summary
        }
        with open(os.path.join(output_folder_path, str(idx) + ".json"), "a", encoding="utf-8") as file:
            json.dump(line_dict, file)
            file.write("\n")

    print("Done with Step 1 for Category: " + category)


def process_academy(input_folder, output_folder, category):
    input_folder_path = os.path.join(input_folder, category)
    output_folder_path = os.path.join(output_folder, category)
    pathlib.Path(output_folder_path).mkdir(parents=True, exist_ok=True)
    for f in os.listdir(output_folder_path):
        os.remove(os.path.join(output_folder_path, f))

    # Step 1
    file_names = os.listdir(input_folder_path)
    print(len(file_names))
    for i, file_name in enumerate(file_names):
        title = ".".join(file_name.split(".")[:-1])
        if 50 <= int(title[:2]) <= 79 and int(title[:2]) != 61:
            idx = 0
        elif 83 <= int(title[:2]) <= 89:
            idx = 1
        else:
            idx = 2
        wiki_page = wikipedia.page(title, auto_suggest=False)
        wiki_title = wiki_page.title
        wiki_summary = wiki_page.content.split("\n")[0].strip()
        line_dict = {
            "title": wiki_title,
            "summary": wiki_summary
        }
        with open(os.path.join(output_folder_path, str(idx) + ".json"), "a", encoding="utf-8") as file:
            json.dump(line_dict, file)
            file.write("\n")

    print("Done with Step 1 for Category: " + category)


def process_music(input_folder, output_folder, category):
    input_folder_path = os.path.join(input_folder, category)
    output_folder_path = os.path.join(output_folder, category)
    pathlib.Path(output_folder_path).mkdir(parents=True, exist_ok=True)
    for f in os.listdir(output_folder_path):
        os.remove(os.path.join(output_folder_path, f))

    # Step 1
    file_names = os.listdir(input_folder_path)
    print(len(file_names))
    for i, file_name in enumerate(file_names):
        title = ".".join(file_name.split(".")[:-1])
        if int(title[:2]) <= 49:
            idx = 0
        else:
            idx = 1
        wiki_page = wikipedia.page(title, auto_suggest=False)
        wiki_title = wiki_page.title
        wiki_summary = wiki_page.content.split("\n")[0].strip()
        line_dict = {
            "title": wiki_title,
            "summary": wiki_summary
        }
        with open(os.path.join(output_folder_path, str(idx) + ".json"), "a", encoding="utf-8") as file:
            json.dump(line_dict, file)
            file.write("\n")

    print("Done with Step 1 for Category: " + category)


def get_topic_pairs(input_folder, output_folder):
    selection_ratio = 0.5

    categories = os.listdir(input_folder)
    print(categories)
    pathlib.Path(output_folder).mkdir(parents=True, exist_ok=True)
    for f in os.listdir(output_folder):
        os.remove(os.path.join(output_folder, f))

    for category in categories:
        category_folder = os.path.join(input_folder, category)
        file_names = os.listdir(category_folder)
        print(file_names)
        for file_name in file_names:
            file_path = os.path.join(category_folder, file_name)
            output_file_path = os.path.join(output_folder, "_".join(category.split(" ")) + "-" + file_name)
            print(file_path)
            print(output_file_path)

            # Read topics and texts
            dataset = []
            with open(file_path, "r", encoding="utf-8") as file:
                raw_lines = file.readlines()
                for raw_line in raw_lines:
                    line = json.loads(raw_line)
                    dataset.append(line)
            n_dataset = len(dataset)
            print(str(n_dataset) + " --> " + str(n_dataset * (n_dataset - 1) / 2))

            # Calculate similarity
            model = AutoModel.from_pretrained('jinaai/jina-embeddings-v2-base-en', trust_remote_code=True)
            embeddings = model.encode([data["summary"] for data in dataset])

            cos_sim = embeddings @ embeddings.T
            norms = norm(embeddings, axis=1)
            for i in range(n_dataset):
                for j in range(n_dataset):
                    if i >= j:
                        cos_sim[i, j] = 0.0
                    cos_sim[i, j] = cos_sim[i, j] / norms[i] / norms[j]
            sim_values = sorted(np.extract(cos_sim > 0, cos_sim), reverse=True)
            sim_threshold = sim_values[int(len(sim_values) * selection_ratio)]

            # Get Top Similar Pairs
            top_sim_pairs = []
            for i in range(n_dataset):
                for j in range(i + 1, n_dataset):
                    if cos_sim[i, j] > sim_threshold:
                        assert (i, j) not in top_sim_pairs
                        assert (j, i) not in top_sim_pairs
                        top_sim_pairs.append((i, j))

            # Get Topic Pairs
            for pair in top_sim_pairs:
                i = pair[0]
                j = pair[1]
                if len(nltk.sent_tokenize(dataset[i]["summary"])) > len(nltk.sent_tokenize(dataset[j]["summary"])):
                    source = dataset[i]
                    target = dataset[j]
                elif len(nltk.sent_tokenize(dataset[i]["summary"])) < len(nltk.sent_tokenize(dataset[j]["summary"])):
                    source = dataset[j]
                    target = dataset[i]
                elif len(nltk.word_tokenize(dataset[i]["summary"])) > len(nltk.word_tokenize(dataset[j]["summary"])):
                    source = dataset[i]
                    target = dataset[j]
                else:
                    source = dataset[j]
                    target = dataset[i]
                line_dict = {
                    "source_topic": source["title"],
                    "source_text": source["summary"],
                    "target_topic": target["title"],
                    "target_text": target["summary"]
                }
                with open(output_file_path, "a", encoding="utf-8") as file:
                    json.dump(line_dict, file)
                    file.write("\n")
            print("Save " + str(len(top_sim_pairs)) + " pairs")
        print("Done with Step 2 for Category: " + category)
        print("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='USNews Data Collection')
    parser.add_argument('--step', type=int)
    args = parser.parse_args()

    step = int(args.step)
    assert step in [1, 2, 3]

    if step == 0:
        categories = [
            "satellite launch",
            "academy award ceremony",
            "music award ceremony",
        ]
        input_folder = "./RoleEE/0_RoleEE_data"
        output_folder = "./RoleEE/0_RoleEE_data_clean"
        # process_satellite(input_folder=input_folder, output_folder=output_folder, category=categories[0])
        # process_academy(input_folder=input_folder, output_folder=output_folder, category=categories[1])
        # process_music(input_folder=input_folder, output_folder=output_folder, category=categories[2])

    # Step 1: get topic/text pairs and mix them
    if step == 1:
        input_folder = "./RoleEE/0_RoleEE_data_clean"
        output_folder = "./RoleEE/1_RoleEE_topic_pair"
        get_topic_pairs(input_folder=input_folder, output_folder=output_folder)

        input_folder = "./RoleEE/1_RoleEE_topic_pair"
        output_file = "./RoleEE/topic_text_pair.json"
        mix_topic_pairs(input_folder=input_folder, output_file=output_file)

    # Step 2: get knowledge pieces
    if step == 2:
        input_file = "./RoleEE/topic_text_pair.json"
        output_file = "./RoleEE/knowledge_pieces.txt"
        construct_knowledge_source(input_file=input_file, output_file=output_file)

    # Step 3: get knowledge piece tensors
    if step == 3:
        input_file = './RoleEE/knowledge_pieces.txt'
        output_folder = './RoleEE/3_knowledge_tensors'
        save_knowledge_tensors(input_file=input_file, output_folder=output_folder)

        input_folder = './RoleEE/3_knowledge_tensors'
        output_file = './RoleEE/knowledge_tensors.pt'
        obtain_knowledge_tensors(input_folder=input_folder, output_file=output_file)
