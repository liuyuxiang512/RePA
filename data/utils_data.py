from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
import torch
import os
import json
import random
import nltk
import pathlib

device = "cuda:0"


def mix_topic_pairs(input_folder, output_file):
    file_names = os.listdir(input_folder)
    print(file_names)

    dataset = []
    for file_name in file_names:
        with open(os.path.join(input_folder, file_name), "r", encoding="utf-8") as file:
            raw_lines = file.readlines()
            for raw_line in raw_lines:
                line = json.loads(raw_line)
                dataset.append(line)
    print("Save " + str(len(dataset)) + " pairs")

    random.seed(4202)
    random.shuffle(dataset)
    with open(output_file, "w", encoding="utf-8") as file:
        for data in dataset:
            json.dump(data, file)
            file.write("\n")

    for f in os.listdir(input_folder):
        os.remove(os.path.join(input_folder, f))
    os.rmdir(input_folder)
    print("Remove " + input_folder)


def construct_knowledge_source(input_file, output_file):
    dataset = []
    with open(input_file, "r", encoding="utf-8") as file:
        raw_lines = file.readlines()
        for raw_line in raw_lines:
            dataset.append(json.loads(raw_line))

    knowledge = []
    ct = 0
    for data in dataset:
        source_topic = data["source_topic"]
        target_topic = data["target_topic"]
        source_text = data["source_text"]
        target_text = data["target_text"]

        source_segments = nltk.sent_tokenize(source_text)
        target_segments = nltk.sent_tokenize(target_text)

        ct += len(source_segments)
        ct += len(target_segments)

        for segment in source_segments:
            if source_topic + ": " + segment in knowledge:
                continue
            knowledge.append(source_topic + ": " + segment)
        for segment in target_segments:
            if target_topic + ": " + segment in knowledge:
                continue
            knowledge.append(target_topic + ": " + segment)

    random.seed(4202)
    for i in range(3):
        knowledge = sorted(knowledge, key=lambda x: random.random())

    with open(output_file, "w", encoding="utf-8") as file:
        file.write("Knowledge Piece:")
        for piece in knowledge:
            file.write("\n" + piece)

    print("Save " + str(len(knowledge)) + " samples out of " + str(ct) + " segments.")


def save_knowledge_tensors(input_file, output_folder):
    pathlib.Path(output_folder).mkdir(parents=True, exist_ok=True)
    for f in os.listdir(output_folder):
        os.remove(os.path.join(output_folder, f))

    pieces_list = []
    with open(input_file, "rb") as file:
        raw_lines = file.readlines()
        for line in raw_lines[1:]:
            pieces_list.append(line.decode().strip())

    print("Read " + str(len(pieces_list)) + " pieces of knowledge")

    contexts = pieces_list

    context_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-multiset-base")
    context_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-multiset-base").to(device)

    division = 100
    for i in range(int(len(contexts) / division) + 1):
        start = i * division
        if start == len(contexts):
            break
        end = min((i + 1) * division, len(contexts))

        input_dict = context_tokenizer(contexts[start: end],
                                       padding='max_length',
                                       max_length=128,
                                       truncation=True,
                                       return_tensors="pt").to(device)
        del input_dict["token_type_ids"]
        encoded_contexts = context_encoder(**input_dict)['pooler_output']  # division x 768

        filename = "piece_start_" + str(start) + "_end_" + str((i + 1) * division - 1) + ".pt"
        filepath = os.path.join(output_folder, filename)

        torch.save(encoded_contexts, filepath)

    print("Save " + str(int(len(contexts) / division) + 1) + " files, " + str(len(contexts)) + " tensors")


def obtain_knowledge_tensors(input_folder, output_file):
    filenames = os.listdir(input_folder)

    device = torch.load(os.path.join(input_folder, filenames[0])).device
    knowledge_tensors = torch.empty(0, 768).to(device)
    print(knowledge_tensors)

    for i in range(len(filenames)):
        filename = filenames[i]
        contexts_tensors = torch.load(os.path.join(input_folder, filename))
        knowledge_tensors = torch.cat((knowledge_tensors, contexts_tensors), 0)

    torch.save(knowledge_tensors, output_file)

    print("Save " + str(len(knowledge_tensors)) + " tensors into " + output_file)
    print(knowledge_tensors.size())

    for f in os.listdir(input_folder):
        os.remove(os.path.join(input_folder, f))
    os.rmdir(input_folder)
    print("Remove " + input_folder)

