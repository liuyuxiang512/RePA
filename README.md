# *Writing Like the Best*: Exemplar-Based Expository Text Generation

---

This repository contains source code and data for [*Writing Like the Best*: Exemplar-Based Expository Text Generation](https://google.com).

## How to Run

### Step 1: Fill in your secret keys in "secret_keys.sh".

### Step 2: Run main experiments

You may change the *Wikipedia* to *RoleEE* or *USNews* for corresponding datasets, or change *GPT4* to *LLaMA3* for different backbone LLM, and change *false* to *true* for debug mode. You may change the corresponding config files for detailed settings, which is under the "configs" folder.

> bash experiments.sh Wikipedia GPT4 false

### Step 3: Run Evaluations

You may change *baseline* to *ablation* for different evaluation modes.

> bash evaluations.sh

## Data

## Reference

If you use our work, please consider citing our preprint:

