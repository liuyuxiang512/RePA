# venv: IW, python=3.10.13
# step 1
import nltk
import requests
from bs4 import BeautifulSoup
import argparse
import random
import json
import os
import pathlib

from transformers import AutoModel
from numpy.linalg import norm
import numpy as np

from utils_data import mix_topic_pairs, construct_knowledge_source, save_knowledge_tensors, obtain_knowledge_tensors

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:121.0) Gecko/20100101 Firefox/121.0'
}


def crawl_colleges(input_file, output_file):
    def fetch_college_urls(input_file):
        with open(input_file, "r") as file:
            html_text = file.read()
        soup = BeautifulSoup(html_text, "html.parser").body.main.contents[3].article.div.contents[4].div.div.div.contents[1].ol.contents
        print("Raw URLs: " + str(len(soup)))

        college_urls = []
        for li in soup:
            try:
                college_url = li.div.contents[1].div.div.div.a["href"]
                college_urls.append(college_url)
            except:
                continue
        print("Obtain URLs: " + str(len(college_urls)))

        return college_urls

    def fetch_college_title_overview(url):
        url = "https://www.usnews.com" + url
        try:
            resp = requests.get(url, headers=HEADERS)
            soup_main = BeautifulSoup(resp.text, "html.parser").body.main

            title = soup_main.contents[2].contents[4].div.div.contents[1].contents[1].div.contents[1].contents[1].h1.text

            paragraphs = []
            soup = soup_main.contents[2].contents[5].div.div.div.div.section.contents[1].div.div.div.div.p.contents
            for p in soup:
                paragraph = p.text
                if paragraph != "":
                    paragraphs.append(paragraph)

            college = {
                "title": title,
                "overview": paragraphs,
            }
            return college
        except:
            return {}

    college_urls = fetch_college_urls(input_file=input_file)
    # Clear the output file
    with open(output_file, "w", encoding="utf-8") as file:
        file.write("")
    # Save
    ct = 0
    for url in college_urls:
        college = fetch_college_title_overview(url=url)
        if not college:
            print(url)
            continue
        with open(output_file, "a", encoding="utf-8") as file:
            json.dump(college, file)
            file.write("\n")
            ct += 1
    print("Save " + str(ct) + " Colleges")


def get_topic_groups(input_file, output_folder):
    pathlib.Path(output_folder).mkdir(parents=True, exist_ok=True)
    for f in os.listdir(output_folder):
        os.remove(os.path.join(output_folder, f))

    dataset = []
    with open(input_file, "r", encoding="utf-8") as file:
        raw_lines = file.readlines()
        for raw_line in raw_lines:
            line = json.loads(raw_line)
            dataset.append(line)
    print("Read " + str(len(dataset)) + " Colleges")

    groups = [[], []]
    for data in dataset:
        if len(data["overview"]) == 1:
            groups[0].append(data)
        else:
            groups[1].append(data)
    print("Group with 1 paragraph of overview: " + str(len(groups[0])))
    print("Group with more paragraph of overview: " + str(len(groups[1])))

    for id in range(len(groups)):
        group = groups[id]
        filename = os.path.join(output_folder, str(id) + ".json")
        with open(filename, "w", encoding="utf-8") as file:
            for data in group:
                json.dump(data, file)
                file.write("\n")


def get_topic_pairs(input_folder, output_folder):
    def process_overview(data):
        overview = data["overview"]
        if len(overview) == 1:
            return overview[0]

        processed_overview = overview[0]
        processed_overview += " "
        processed_overview += nltk.sent_tokenize(overview[1])[0]
        processed_overview = processed_overview.strip()
        return processed_overview

    pathlib.Path(output_folder).mkdir(parents=True, exist_ok=True)
    for f in os.listdir(output_folder):
        os.remove(os.path.join(output_folder, f))

    filenames = os.listdir(input_folder)
    for filename in filenames:
        filepath = os.path.join(input_folder, filename)

        dataset = []
        with open(filepath, "r", encoding="utf-8") as file:
            raw_lines = file.readlines()
            for raw_line in raw_lines:
                line = json.loads(raw_line)
                line["overview"] = process_overview(line)
                dataset.append(line)

        if filename.startswith("0"):
            top_sim_pairs = [(i, i + 1) for i in range(150)]
            top_sim_pairs += [(i, i + 2)for i in range(100)]
            print(filename)
            print("Get " + str(len(top_sim_pairs)) + " pairs from " + str(len(dataset)) + " colleges")
        else:
            # Calculate similarity
            model = AutoModel.from_pretrained('jinaai/jina-embeddings-v2-base-en', trust_remote_code=True)
            embeddings = model.encode([nltk.sent_tokenize(data["overview"])[-1] for data in dataset])

            cos_sim = embeddings @ embeddings.T
            norms = norm(embeddings, axis=1)
            for i in range(len(dataset)):
                for j in range(len(dataset)):
                    if i >= j:
                        cos_sim[i, j] = 0.0
                    cos_sim[i, j] = cos_sim[i, j] / norms[i] / norms[j]
            sim_values = sorted(np.extract(cos_sim > 0, cos_sim), reverse=True)
            sim_threshold = sim_values[250]

            # Get Top Similar Pairs
            top_sim_pairs = []
            for i in range(len(dataset)):
                for j in range(i + 1, len(dataset)):
                    if cos_sim[i, j] > sim_threshold:
                        assert (i, j) not in top_sim_pairs
                        assert (j, i) not in top_sim_pairs
                        top_sim_pairs.append((i, j))
            print(filename)
            print("Get " + str(len(top_sim_pairs)) + " pairs from " + str(len(dataset)) + " colleges")

        # Get College Pairs
        for pair in top_sim_pairs:
            i = pair[0]
            j = pair[1]
            if len(nltk.sent_tokenize(dataset[i]["overview"])) > len(nltk.sent_tokenize(dataset[j]["overview"])):
                source = dataset[i]
                target = dataset[j]
            elif len(nltk.sent_tokenize(dataset[i]["overview"])) < len(nltk.sent_tokenize(dataset[j]["overview"])):
                source = dataset[j]
                target = dataset[i]
            elif len(nltk.word_tokenize(dataset[i]["overview"])) > len(nltk.word_tokenize(dataset[j]["overview"])):
                source = dataset[i]
                target = dataset[j]
            else:
                source = dataset[j]
                target = dataset[i]
            line_dict = {
                "source_topic": source["title"],
                "source_text": source["overview"],
                "target_topic": target["title"],
                "target_text": target["overview"]
            }
            with open(os.path.join(output_folder, filename), "a", encoding="utf-8") as file:
                json.dump(line_dict, file)
                file.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='USNews Data Collection')
    parser.add_argument('--step', type=int)
    args = parser.parse_args()

    step = int(args.step)
    assert step in [0, 1, 2, 3]

    # Step 0: Crawl USNews Colleges and Group Colleges
    if step == 0:
        input_file = './USNews/0_USNews_html'
        output_file = './USNews/0_USNews_title_overview.json'
        crawl_colleges(input_file=input_file, output_file=output_file)

        input_file = './USNews/0_USNews_title_overview.json'
        output_folder = './USNews/0_USNews_colleges'
        get_topic_groups(input_file=input_file, output_folder=output_folder)

    # Step 1: get topic/text pairs and mix them
    if step == 1:
        input_folder = './USNews/0_USNews_colleges'
        output_folder = './USNews/1_USNews_college_pair'
        get_topic_pairs(input_folder=input_folder, output_folder=output_folder)

        input_folder = './USNews/1_USNews_college_pair'
        output_file = './USNews/topic_text_pair.json'
        mix_topic_pairs(input_folder=input_folder, output_file=output_file)

    # Step 2: get knowledge pieces
    if step == 2:
        input_file = "./USNews/topic_text_pair.json"
        output_file = "./USNews/knowledge_pieces.txt"
        construct_knowledge_source(input_file=input_file, output_file=output_file)

    # Step 3: get knowledge piece tensors
    if step == 3:
        input_file = './USNews/knowledge_pieces.txt'
        output_folder = './USNews/3_knowledge_tensors'
        save_knowledge_tensors(input_file=input_file, output_folder=output_folder)

        input_folder = './USNews/3_knowledge_tensors'
        output_file = './USNews/knowledge_tensors.pt'
        obtain_knowledge_tensors(input_folder=input_folder, output_file=output_file)
