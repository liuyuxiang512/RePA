from openai import OpenAI
from diskcache import Cache
import logging
import json
import replicate

logging.basicConfig(level=logging.INFO)


def read_dataset(input_file):
    # read json file, each line is a dict of a data sample
    dataset = []
    with open(input_file, "r", encoding="utf-8") as file:
        raw_lines = file.readlines()
        for raw_line in raw_lines:
            dataset.append(json.loads(raw_line))

    return dataset


def get_api_response(content: str, llm_kwargs: dict):
    model = llm_kwargs["model"]

    if model.startswith("gpt-4") or model.startswith("o1-preview"):
        cache = Cache(".openai_" + model + "_cache")
        cache_key = json.dumps(llm_kwargs) + "\n\n\n\n\n" + content

        if cache_key in cache:
            log_info = content.split("\n")[0].split(" ")[0]
            logging.info(f'OpenAI CACHE: {log_info}')

            return cache[cache_key]
        client = OpenAI(api_key=llm_kwargs["OPENAI_API_KEY"])
        chat_completion = client.chat.completions.create(
            model=llm_kwargs["model"],
            messages=[
                {
                    "role": "user",
                    "content": content,
                }
            ],
            temperature=llm_kwargs["temperature"],
            frequency_penalty=llm_kwargs["frequency_penalty"],
            # logit_bias={"198": -100},
            max_tokens=llm_kwargs["max_generation_tokens"],
            # seed=llm_kwargs["seed"],
            n=llm_kwargs["n_choices"],
        )
        response = chat_completion.choices[0].message.content
        cache[cache_key] = response
        return response
    # elif "llama-3-70b" in model:  ## starts with llama3
    else:
        cache = Cache(".replicate_" + model + "_cache")
        cache_key = json.dumps(llm_kwargs) + "\n\n\n\n\n" + content

        if cache_key in cache:
            log_info = content.split("\n")[0].split(" ")[0]
            logging.info(f'Replicate CACHE: {log_info}')
            return cache[cache_key]
        client = replicate.Client(api_token=llm_kwargs["replicate_api_token"])
        response = client.run(
            model,
            input={
                "top_k": 0,
                "top_p": 0.9,
                "prompt": content,
                "temperature": llm_kwargs["temperature"],
                # "system_prompt": "You are a helpful assistant",
                "length_penalty": 1,
                "max_new_tokens": llm_kwargs["max_generation_tokens"],
                "stop_sequences": "<|end_of_text|>,<|eot_id|>",
                "prompt_template": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
                "presence_penalty": 0
            }
        )
        response = "".join(response)
        cache[cache_key] = response
        return response