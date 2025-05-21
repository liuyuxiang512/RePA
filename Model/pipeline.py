import nltk
from utils import get_api_response
import os
from Model.Retriever.retrieve import Retriever
import logging
from pprint import pprint

logging.basicConfig(level=logging.INFO)


class WritingAssistant:
    def __init__(self, data_sample, prompts_folder, llm_kwargs, retriever_kwargs, ablation, debug=True):
        # Fixed attributes
        self.source_topic = data_sample["source_topic"]
        self.source_text = data_sample["source_text"]
        self.target_topic = data_sample["target_topic"]
        self.target_text = data_sample["target_text"]

        # Fixed settings
        self.ablation = ablation
        self.debug = debug
        self.llm_kwargs = llm_kwargs
        self.retriever_kwargs = retriever_kwargs

        # Fixed prompts
        self.prompts = {'clarify': None, 'outline': None, 'transfer': None, "calibratedQA": None, "write": None,
                        "summarize": None}
        for key in self.prompts.keys():
            with open(os.path.join(prompts_folder, "prompt_" + key + ".txt"), "r") as file:
                self.prompts[key] = file.read()

        # Running texts
        self.source_segments = None
        self.segment = None

        self.clarified_segment = None
        self.valid_sources = ""
        self.source_queries = None
        self.target_queries = None
        self.knowledge = None
        self.target_answers = []
        self.draft_segment = None
        self.target_segment = None
        self.output_text = ""

        # Running memory
        self.short_memory = [self.source_topic + ":"]
        self.long_memory = ""

        # For logging
        self.logging_info = {
            "source topic": self.source_topic,
            "source text": self.source_text,
            "target topic": self.target_topic,
            "target text": self.target_text
        }

    def call_segment(self):
        """
        Input:
            self.source_text
        Output:
            self.source_segments
        """
        # nltk.download('punkt')
        self.source_segments = nltk.sent_tokenize(self.source_text)

    def _construct_context(self, phase, idx_query=None):
        user_text = None

        if phase == "clarify":
            user_text = self.prompts["clarify"]
            user_text = user_text.replace("${SOURCE_TOPIC}", self.source_topic)
            user_text = user_text.replace("${SHORT_TERM_MEMORY}", " ".join(self.short_memory))
            user_text = user_text.replace("${SEGMENT}", self.segment)

        if phase == "outline":
            user_text = self.prompts["outline"]
            user_text = user_text.replace("${SOURCE_TOPIC}", self.source_topic)
            user_text = user_text.replace("${CLARIFIED_SEGMENT}", self.clarified_segment)
            user_text = user_text.replace("${ERROR_TOKEN}", self.llm_kwargs["outline_error_token"])

        if phase == "transfer":
            user_text = self.prompts["transfer"]
            user_text = user_text.replace("${SOURCE_TOPIC}", self.source_topic)
            user_text = user_text.replace("${TARGET_TOPIC}", self.target_topic)
            user_text = user_text.replace("${SOURCE_QUERY}", self.source_queries[idx_query])

        if phase == "calibratedQA":
            user_text = self.prompts["calibratedQA"]
            user_text = user_text.replace("${TARGET_TOPIC}", self.target_topic)
            user_text = user_text.replace("${TARGET_QUERY}", self.target_queries[idx_query])
            user_text = user_text.replace("${ERROR_TOKEN}", self.llm_kwargs["calibratedQA_error_token"])
            knowledge = self._process_knowledge(idx_query=idx_query)
            user_text = user_text.replace("${KNOWLEDGE}", knowledge)

        if phase == "write":
            user_text = self.prompts["write"]
            user_text = user_text.replace("${TARGET_TOPIC}", self.target_topic)
            user_text = user_text.replace("${SEGMENT}", self.segment)
            user_text = user_text.replace("${TARGET_ANSWERS}", " ".join(self.target_answers))
            user_text = user_text.replace("${LONG_TERM_MEMORY}", self.long_memory)

        if phase == "summarize":
            user_text = self.prompts["summarize"]
            user_text = user_text.replace("${TARGET_TOPIC}", self.target_topic)
            user_text = user_text.replace("${TARGET_SEGMENT}", self.target_segment)
            user_text = user_text.replace("${LONG_TERM_MEMORY}", self.long_memory)

        return user_text

    def _process_knowledge(self, idx_query):
        retrievers = self.retriever_kwargs["retrievers"]
        knowledge_all = []
        for retriever in retrievers:
            if retriever == "Bing":
                knowledge = self.knowledge["Bing"]
                knowledge = knowledge[:min(self.llm_kwargs["num_knowledge_piece"]["Bing"], len(knowledge))]
                knowledge = ['Title: ' + piece['name'] + '\nSnippet: ' + piece['snippet'] for piece in knowledge]
                knowledge = '\n\n'.join(knowledge)
                knowledge_all.append(knowledge)
            elif retriever == "DPR":
                knowledge = self.knowledge["DPR"][idx_query]
                knowledge = knowledge[:min(self.llm_kwargs["num_knowledge_piece"]["DPR"], len(knowledge))]
                knowledge = '\n\n'.join(knowledge)
                knowledge_all.append(knowledge)
            else:
                exit("Retriever is wrong")

        return '\n\n'.join(knowledge_all)

    def call_clarify(self):
        """
        Input:
            self.segment,
            self.short_memory
        Output:
            self.clarified_segment,
            self.short_memory
        """

        if self.ablation == "woClarify":
            logging.info("Ablation: " + self.ablation)
            self.clarified_segment = self.segment
        else:
            user_text = self._construct_context(phase="clarify")
            response = get_api_response(content=user_text, llm_kwargs=self.llm_kwargs)

            # Process response
            responses = response.split("\n\n")
            for response_line in responses:
                if response_line.startswith("Clarified TEXT:"):
                    response = response_line.replace("Clarified TEXT:", "").strip()
                    break

            self.clarified_segment = response.strip()

            # Update short-term memory as latest 3 clarified segments
            self.short_memory.append(self.clarified_segment)
            if len(self.short_memory) > 3:
                self.short_memory = self.short_memory[-3:]

    def call_outline(self):
        """
        Input:
            self.clarified_segment
        Mid:
            self.source_queries: might be empty []
        Output:
            self.target_queries: might be empty []
        """

        if self.ablation == "woOutline":
            logging.info("Ablation: " + self.ablation)
            self.source_queries = [self.clarified_segment]
        else:
            # Step 1: question generation
            user_text = self._construct_context(phase="outline")
            response = get_api_response(content=user_text, llm_kwargs=self.llm_kwargs)

            # Process response
            if self.llm_kwargs["outline_error_token"] in response:
                self.source_queries = []
            else:
                self.source_queries = []
                for r in response.split("\n"):
                    if r.startswith("Q:") or r.strip().endswith("?"):
                        self.source_queries.append(r.replace("Q:","").strip())

        if not self.source_queries:
            self.target_queries = []
            logging.info("=== Transfer ===")
            print("... No Source Queries ...")
            print("\n")
            return

        # Step 2: question transfer
        self.target_queries = []
        for idx_query in range(len(self.source_queries)):
            # Query Transfer one by one
            source_query = self.source_queries[idx_query]
            if self.source_topic in source_query:
                target_query = source_query.replace(self.source_topic, self.target_topic)
            else:
                user_text = self._construct_context(phase="transfer", idx_query=idx_query)
                response = get_api_response(content=user_text, llm_kwargs=self.llm_kwargs)
                if "llama" in self.llm_kwargs["model"]:
                    original_response = response
                    response = response.split("\n\n")[1].split("\n")[0]
                    if self.ablation != "woOutline":
                        if response.endswith("?"):
                            continue
                        else:
                            response = ""
                            for piece in original_response.split("\n\n"):
                                if piece.endswith("?"):
                                    response = piece
                        assert response.endswith("?"), response
                # process response
                target_query = response.strip()
            self.target_queries.append(target_query)

    def call_calibrated_qa(self):
        """
        Input:
            self.target_queries,
        Output:
            self.knowledge
        Output:
            self.target_answers
        """
        if not self.target_queries:
            logging.info("=== Calibrated QA ===")
            print("... No Target Queries ...")
            print("\n")
            return

        # Retrieve in batch
        retriever = Retriever(self.retriever_kwargs)
        self.knowledge = retriever.retrieve(topic=self.target_topic, queries=self.target_queries)

        if self.ablation == "woOutline":
            knowledge = self._process_knowledge(idx_query=0)
            self.target_answers = knowledge.split("\n\n")
        else:
            self.target_answers = []
            for idx_query in range(len(self.target_queries)):
                # Question Answering
                user_text = self._construct_context(phase="calibratedQA", idx_query=idx_query)
                response = get_api_response(content=user_text, llm_kwargs=self.llm_kwargs)

                if self.llm_kwargs["calibratedQA_error_token"] in response:
                    continue
                if "\n\nConfidence Probability:" not in response:
                    response = response.replace("\nConfidence Probability:", "\n\nConfidence Probability:")
                    response = response.replace("idenceConf", "Confidence")
                if "Response Text Mis:" in response:
                    response = response.replace("Response Text Mis:", "Response Text: Mis")
                assert response.split("\n\n")[0].startswith("Response Text:"), response
                assert response.split("\n\n")[1].startswith("Confidence Probability:"), response
                answer = response.split("\n\n")[0].replace("Response Text:", "").strip()
                confidence = response.split("\n\n")[1].replace("Confidence Probability:", "").strip().split(" ")[0]
                confidence = float(confidence)

                if self.ablation == "woRefusal":
                    logging.info("Ablation: " + self.ablation)
                    self.target_answers.append(answer)
                else:
                    if confidence >= self.llm_kwargs["confidence_threshold"]:
                        if self.llm_kwargs["calibratedQA_error_token"] not in answer:
                            self.target_answers.append(answer)

    def call_write(self):
        """
        Input:
            self.target_topic,
            self.target_answers,
            self.segment,

            self.output_text,
            self.long_memory
        Output:
            self.draft_segment,
            self.target_segment,

            self.output_text,
            self.long_memory
        """
        if not self.target_answers:
            logging.info("=== Write ===")
            print("... No Answers ...")
            print("\n")
            return

        # Stage 1: Write
        user_text = self._construct_context(phase="write")
        response = get_api_response(content=user_text, llm_kwargs=self.llm_kwargs)

        response = response.replace("SEGENT", "SEGMENT")
        response = response.replace("**", "")
        response = response.replace(".\nFINAL SEGMENT:", ".\n\nFINAL SEGMENT:")
        response = response.replace(".\nDRAFT SEGMENT:", ".\n\nFINAL SEGMENT:")
        self.draft_segment = None
        self.target_segment = None
        for r in response.split("\n\n"):
            r = r.strip()
            if r.startswith("DRAFT SEGMENT:"):
                self.draft_segment = r.replace("DRAFT SEGMENT:", "").strip()
            if r.startswith("FINAL SEGMENT:"):
                self.target_segment = r.replace("FINAL SEGMENT:", "").strip()
        if not self.draft_segment:
            return

        if not self.target_segment:
            self.target_segment = self.draft_segment

        self.valid_sources += "\n" + self.clarified_segment

        assert self.draft_segment
        assert self.target_segment
        if self.target_segment == "N/A":
            logging.info("=== Write ===")
            print("... No Segment ...")
            print("\n")
            return

        # Stage 2: Summarize
        if self.ablation == "woRevise":
            logging.info("Ablation: " + self.ablation)
            self.target_segment = self.draft_segment
            self.long_memory = ""
        else:
            user_text = self._construct_context(phase="summarize")
            response = get_api_response(content=user_text, llm_kwargs=self.llm_kwargs)
            # process response
            if "llama" in self.llm_kwargs["model"]:
                try:
                    response = response.split("\n\n")[1]
                except:
                    response = response.split("\n")[1]
            self.long_memory = response.strip()

        # Stage 3
        self.output_text += " " + self.target_segment

    def get_assistant_response(self):
        self.call_segment()

        for i in range(len(self.source_segments)):
            self.segment = self.source_segments[i]

            self.call_clarify()

            self.call_outline()

            self.call_calibrated_qa()

            self.call_write()

        self.logging_info["output text"] = self.output_text

        if self.debug:
            logging.info("=== Results ===")
            print(self.source_topic)
            print(self.source_text)
            print("\n")
            print(self.target_topic)
            print(self.target_text)
            print("\n")
            print("Valid Source:")
            print(self.valid_sources)
            print("Output:")
            print(self.output_text)
            print("\n")

    def return_logging_into(self):
        return self.logging_info
