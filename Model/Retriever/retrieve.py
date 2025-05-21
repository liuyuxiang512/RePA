from Model.Retriever.bing_search import search_bing_batch
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
import torch


class Retriever:
    def __init__(self, retriever_kwargs):
        self.engines = retriever_kwargs["retrievers"]
        self.kwargs = retriever_kwargs

    def retrieve(self, topic: str, queries: list[str]):
        retrieved_results = {}
        for engine in self.engines:
            if engine == "Bing":
                retriever = SearchEngine(self.kwargs)
                retrieved_results[engine] = retriever.retrieve(topic=topic)
            elif engine == "DPR":
                retriever = DPR(self.kwargs)
                retrieved_results[engine] = retriever.retrieve(queries=queries)

        return retrieved_results


class SearchEngine:
    def __init__(self, kwargs):
        self.engine = kwargs["retrievers"]
        assert "Bing" in self.engine
        self.kwargs = kwargs

    def retrieve(self, topic: str):
        # Search Engine is used per topic
        topic_result = search_bing_batch([topic], self.kwargs)[0]
        return topic_result


class DPR:
    def __init__(self, kwargs):
        self.engine = kwargs["retrievers"]
        assert "DPR" in self.engine
        self.kwargs = kwargs
        self.topk = self.kwargs["topk_DPR"]

        # Obtain knowledge as well as their encodings
        self.knowledge = []
        with open(self.kwargs["DPR_knowledge_pieces"], "rb") as file:
            raw_lines = file.readlines()
            for line in raw_lines[1:]:
                self.knowledge.append(line.decode().strip())

        self.encoded_knowledge = torch.load(self.kwargs["DPR_knowledge_tensors"])
        self.device = self.encoded_knowledge.device

        self.question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-multiset-base")
        self.question_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-multiset-base").to(self.device)

    def retrieve(self, queries: list[str]):
        # DPR is used per query
        queries_results = []

        for question in queries:
            encoded_question = self.encode_question(question)
            scores = torch.mm(encoded_question, self.encoded_knowledge.T)

            if self.topk >= len(self.knowledge):
                question_results = self.knowledge
            else:
                top_k_results = torch.topk(scores, self.topk).indices.squeeze()
                question_results = [self.knowledge[i] for i in top_k_results]

            queries_results.append(question_results)

        return queries_results

    def encode_question(self, question):
        input_dict = self.question_tokenizer(question,
                                             padding="max_length",
                                             max_length=64,
                                             truncation=True,
                                             return_tensors="pt").to(self.device)
        del input_dict["token_type_ids"]
        encoded_question = self.question_encoder(**input_dict)['pooler_output']  # 1 x 768

        return encoded_question

