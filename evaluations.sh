#!/usr/bin/env bash

source secret_keys.sh

python -m Evaluations.evaluations_basic --mode baseline

python -m Evaluations.evaluations_entailment --mode baseline

python -m Evaluations.evaluations_LLM --mode baseline --openai_key ${OPENAI_API_KEY} --replicate_key ${REPLICATE_API_TOKEN}
