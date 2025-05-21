#!/usr/bin/env bash

debug=true

source secret_keys.sh

dataset=$1
backbone=$2
debug=$3

config_file="./configs/${dataset}_${backbone}_config.json"

if [[ ${debug} == "true" ]]; then
  python -m main \
  --config_file ${config_file} \
  --openai_key ${OPENAI_API_KEY} \
  --azure_key ${BING_SEARCH_V7_SUBSCRIPTION_KEY} \
  --hf_token ${HF_ACCESS_TOKEN} \
  --groq_token ${GROQ_API_TOKEN} \
  --replicate_token ${REPLICATE_API_TOKEN} \
  --debug
  exit
else
  python -m main \
  --config_file ${config_file} \
  --openai_key ${OPENAI_API_KEY} \
  --azure_key ${BING_SEARCH_V7_SUBSCRIPTION_KEY} \
  --hf_token ${HF_ACCESS_TOKEN} \
  --groq_token ${GROQ_API_TOKEN} \
  --replicate_token ${REPLICATE_API_TOKEN}
  exit
fi