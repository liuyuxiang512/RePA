# venv: IW, python=3.10.13
# step 1
from wikipedia2vec import Wikipedia2Vec, dictionary
import pandas as pd
import os
import argparse
import wikipedia
import json
import nltk
import re
import random

from utils_data import mix_topic_pairs, construct_knowledge_source, save_knowledge_tensors, obtain_knowledge_tensors

from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
import torch

"""
https://wikipedia.readthedocs.io/en/latest/code.html#api
"""

similarity_threshold = 0.95
common_threshold = 0.3
embedding_file = "/home/yuxiang/wikipedia/enwiki-20240401-embeddings"


def get_entities(input_file, output_file):
    print("... Reading Wikipedia titles ...")
    entities = []
    with open(input_file, "r") as file:
        data = file.read().strip().split("\n")
        print(data[0])
        print(data[1])
        print(data[-2])
        print(data[-1])
        ct = 0
        for line in data[1:]:
            if bool(re.match('^[_a-zA-Z0-9]*$', line)):
                entities.append(line)
                ct += 1
    print("Number of Entities: " + str(ct))  # 10839144

    print("... Saving filtered entities ...")
    df = pd.DataFrame(entities, columns=['title'])
    df.to_csv(output_file, index=False)
    print("Done!")


def get_individual_pairs(input_file, output_file):
    print("... Reading entity list ...")
    entities = pd.read_csv(input_file).sample(frac=0.2, random_state=42)["title"]
    print(len(entities))  # 6458670 --> 64587; 10839144 --> 2167829.
    print(entities.iloc[:10])

    print("... Loading pre-trained embeddings ...")
    wiki2vec = Wikipedia2Vec.load(embedding_file)

    print("... Read last saved idx ...")
    if not os.path.exists(output_file):
        with open(output_file, "w") as file:
            file.write("idx" + "\t" + "entity" + "\t" + "similar_entity" + "\t" + "similarity" + "\n")
    with open(output_file, "rb") as file:
        try:
            file.seek(-2, os.SEEK_END)
            while file.read(1) != b'\n':
                file.seek(-2, os.SEEK_CUR)
        except OSError:
            file.seek(0)
        last_line = file.readline().decode().strip()
        old_idx = last_line.split("\t")[0]
        print(old_idx)
        if old_idx == "idx":
            new_idx = 0
        else:
            new_idx = int(old_idx) + 1

    print("... Saving from idx " + str(new_idx) + " ...")
    for i in range(new_idx, len(entities)):
        entity = entities.iloc[i]

        if type(entity) is not str:
            continue

        entity = " ".join(entity.split("_"))

        try:
            results = wiki2vec.most_similar(wiki2vec.get_entity(entity), 2)
        except:
            print("cannot find entities")
            continue

        current = results[0][0]
        current_score = results[0][1]
        most_similar = results[1][0]
        similar_score = results[1][1]

        if (type(current) is dictionary.Entity) and (type(most_similar) is dictionary.Entity) and (similar_score > similarity_threshold):
            with open(output_file, "a") as file:
                file.write(str(i) + "\t" + current.title + "\t" + most_similar.title + "\t" + str(similar_score) + "\n")
        else:
            print(entity)
            print(results)
            pass


def get_mutual_pairs(input_file, output_file):
    print("... Reading individual entity pairs ...")
    entity_pairs = []
    with open(input_file, "rb") as file:
        raw_data = file.readlines()
        for line in raw_data[1:]:
            pair = line.decode().strip().split("\t")
            entity_pairs.append([int(pair[0]), pair[1], pair[2], pair[3]])
    print(len(entity_pairs))

    print("... Loading pre-trained embeddings ...")
    wiki2vec = Wikipedia2Vec.load(embedding_file)

    print("... Read last saved idx ...")
    if not os.path.exists(output_file):
        with open(output_file, "w") as file:
            file.write("idx" + "\t" + "idx_1" + "\t" + "entity_1" + "\t" + "entity_2" + "\t" + "similarity" + "\n")
    with open(output_file, "rb") as file:
        try:
            file.seek(-2, os.SEEK_END)
            while file.read(1) != b'\n':
                file.seek(-2, os.SEEK_CUR)
        except OSError:
            file.seek(0)
        last_line = file.readline().decode().strip()
        old_idx = last_line.split("\t")[0]
        print(old_idx)
        if old_idx == "idx":
            new_idx = 0
        else:
            new_idx = int(old_idx) + 1

    print("... Saving from idx " + str(new_idx) + " ...")
    for i in range(new_idx, len(entity_pairs)):
        pair = entity_pairs[i]
        idx_1 = pair[0]
        entity_1 = pair[1]
        entity_2 = pair[2]
        similarity = pair[3]

        most_similar_result = wiki2vec.most_similar(wiki2vec.get_entity(entity_2), 2)[1]
        most_similar = most_similar_result[0]
        similar_score = most_similar_result[1]

        if (type(most_similar) is dictionary.Entity) and similar_score > similarity_threshold:
            with open(output_file, "a") as file:
                file.write(str(i) + "\t" + str(idx_1) + "\t" + entity_1 + "\t" + entity_2 + "\t" + similarity + "\n")
            if most_similar.title != entity_1:
                print(entity_1)
                print(most_similar_result)
                with open(output_file, "a") as file:
                    file.write(str(i) + "\t" + str(idx_1) + "\t" + most_similar.title + "\t" + entity_2 + "\t" + str(similar_score) + "\n")
        else:
            print(pair)


def get_documents(input_file, output_file):
    print("... Reading mutual entity pairs ...")
    entity_pairs = []
    with open(input_file, "rb") as file:
        raw_data = file.readlines()
        for line in raw_data[1:]:
            pair = line.decode().strip().split("\t")
            entity_pairs.append([int(pair[0]), int(pair[1]), pair[2], pair[3]])
    print(len(entity_pairs))

    print("... Read last saved idx ...")
    if not os.path.exists(output_file):
        with open(output_file, "w") as file:
            columns = ["idx", "idx_2", "idx_1", "source_entity", "source_text", "target_entity", "target_text"]
            file.write("\t".join(columns) + "\n")
    with open(output_file, "rb") as file:
        try:
            file.seek(-2, os.SEEK_END)
            while file.read(1) != b'\n':
                file.seek(-2, os.SEEK_CUR)
        except OSError:
            file.seek(0)
        last_line = file.readline().decode().strip()
        old_idx = last_line.split("\t")[0]
        print(old_idx)
        if old_idx == "idx":
            new_idx = 0
        else:
            new_idx = int(old_idx) + 1

    print("... Saving from idx " + str(new_idx) + " ...")
    for i in range(new_idx, len(entity_pairs)):
        pair = entity_pairs[i]

        idx_2 = pair[0]
        idx_1 = pair[1]
        source_entity = pair[2]
        target_entity = pair[3]

        try:
            source_text = wikipedia.page(source_entity).summary
            target_text = wikipedia.page(target_entity).summary
        except:
            print(pair)
            continue

        source_text = "  ".join(source_text.split("\n"))
        target_text = "  ".join(target_text.split("\n"))

        if not len(source_text) or not len(target_text):
            continue

        line = [str(i), str(idx_2), str(idx_1), source_entity, source_text, target_entity, target_text]

        with open(output_file, "a") as file:
            file.write("\t".join(line) + "\n")


def preprocess(input_file, output_file):
    # Clear the output file
    with open(output_file, "w", encoding="utf-8") as file:
        file.write("")

    with open(input_file, "rb") as file:
        raw_lines = file.readlines()

    count = 0
    for raw_line in raw_lines[1:]:
        line = raw_line.decode().strip().split("\t")
        source_topic = line[3]
        source_text = line[4]
        target_topic = line[5]
        target_text = line[6]

        # Preprocess: Filter
        source_segments = nltk.sent_tokenize(source_text)
        target_segments = nltk.sent_tokenize(target_text)
        # too short
        if len(source_segments) < 3 or len(target_segments) < 3:
            continue
        bad_seg_flag = False
        # bad seg: too short
        for source_segment in source_segments:
            if len(source_segment) < 10:
                bad_seg_flag = True
                break
        if bad_seg_flag:
            continue
        # bad seg: start with lowercase
        bad_seg_flag = False
        for source_segment in source_segments:
            if source_segment[0].islower():
                bad_seg_flag = True
                break
        if bad_seg_flag:
            continue
        if len(source_topic) > 30 or len(target_topic) > 30:
            continue
        # bad topic
        if source_topic not in source_segments[0]:
            continue
        # bad pair
        if (len(target_segments) - len(source_segments)) / min(len(source_segments), len(target_segments)) > 1/4:
            continue
        n_source = len(source_text.split(" "))
        n_target = len(target_text.split(" "))
        if (n_source - n_target) / min(n_source, n_target) > 0.3:
            continue
        # short text
        if n_source < 60 or n_target < 60:
            continue
        # long topic name which might cause confusion when retrieval
        if len(source_topic.split(" ")) > 3 or len(target_topic.split(" ")) > 3:
            continue
        # not similar according to their categories
        try:
            categories = [
                wikipedia.page(source_topic).categories,
                wikipedia.page(target_topic).categories
            ]
            commons = []
            for element in categories[0]:
                if element in categories[1]:
                    commons.append(element)
            categories[0] = [category for category in categories[0] if "Wikidata" not in category]
            categories[1] = [category for category in categories[1] if "Wikidata" not in category]
            commons = [category for category in commons if "Wikidata" not in category]
            common_score = len(commons) / (len(categories[0]) + len(categories[1]))
            if common_score < common_threshold:
                continue
        except Exception as ex:
            print(ex)
            continue

        print("Save: " + source_topic + " --> " + target_topic)
        print("\n")

        line_dict = {
            "source_topic": source_topic,
            "source_text": source_text,
            "target_topic": target_topic,
            "target_text": target_text,
        }
        with open(output_file, "a", encoding="utf-8") as file:
            json.dump(line_dict, file)
            file.write("\n")
            count += 1

    print("Save " + str(count) + " samples")


if __name__ == "__main__":
    files = {
        -3: {
            "input_file": './Wikipedia/Wikipedia_data/enwiki-20240401-all-titles-in-ns0',
            "output_file": './Wikipedia/Wikipedia_data/0_enwiki-20240401-all-titles-in-ns0_wo_ambiguity.csv',
        },
        -2: {
            "input_file": './Wikipedia/Wikipedia_data/0_enwiki-20240401-all-titles-in-ns0_wo_ambiguity.csv',
            "output_file": './Wikipedia/Wikipedia_data/1_individual_pairs.txt',
        },
        -1: {
            "input_file": './Wikipedia/Wikipedia_data/1_individual_pairs.txt',
            "output_file": 'Wikipedia/Wikipedia_data/2_mutual_pairs.txt',
        },
        0: {
            "input_file": './Wikipedia/Wikipedia_data/2_mutual_pairs.txt',
            "output_file": './Wikipedia/Wikipedia_data/3_entity_text.txt',
        }
    }

    parser = argparse.ArgumentParser(description='Data Collection')
    parser.add_argument('--step', type=int)
    args = parser.parse_args()

    step = int(args.step)
    assert step in [1, 2, 3]

    if step <= 0:
        input_file = files[step]["input_file"]
        output_file = files[step]["output_file"]
        if step == -3:
            print("... Step -3: Filter Wikipedia Titles ...\n")
            get_entities(input_file=input_file, output_file=output_file)
        if step == -2:
            print("... Step -2: Save Similar Wikipedia Entity Pairs ...\n")
            get_individual_pairs(input_file=input_file, output_file=output_file)
        if step == -1:
            print("... Step -1: Save Similar Wikipedia Entity Pairs ...\n")
            get_mutual_pairs(input_file=input_file, output_file=output_file)
        if step == 0:
            print("... Step 0: Crawl and Save Texts ...\n")
            get_documents(input_file=input_file, output_file=output_file)

    # Step 1: get topic/text pairs -- preprocess and filter
    if step == 1:
        input_file = './Wikipedia/Wikipedia_data/3_entity_text.txt'
        output_file = './Wikipedia/topic_text_pair.json'
        preprocess(input_file=input_file, output_file=output_file)

    # Step 2: get knowledge pieces
    if step == 2:
        input_file = "./Wikipedia/topic_text_pair.json"
        output_file = "./Wikipedia/knowledge_pieces.txt"
        construct_knowledge_source(input_file=input_file, output_file=output_file)

    # Step 3: get knowledge piece tensors
    if step == 3:
        input_file = './Wikipedia/knowledge_pieces.txt'
        output_folder = './Wikipedia/3_knowledge_tensors'
        save_knowledge_tensors(input_file=input_file, output_folder=output_folder)

        input_folder = './Wikipedia/3_knowledge_tensors'
        output_file = './Wikipedia/knowledge_tensors.pt'
        obtain_knowledge_tensors(input_folder=input_folder, output_file=output_file)
